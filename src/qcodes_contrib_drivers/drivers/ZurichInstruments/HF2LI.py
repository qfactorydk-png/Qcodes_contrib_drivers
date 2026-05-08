from typing import Dict, List, Optional, Sequence, Any, Union
from functools import partial
import numpy as np
import logging

from py import process
log = logging.getLogger(__name__)

import time
import matplotlib.pyplot as plt
import numpy as np

import zhinst.utils
import qcodes as qc
from qcodes.instrument.base import Instrument
import qcodes.utils.validators as vals

from qcodes.instrument.parameter import ParameterWithSetpoints, Parameter

from qcodes.dataset.measurements import Measurement, res_type, DataSaver
from qcodes.instrument.specialized_parameters import ElapsedTimeParameter

class HF2LI(Instrument):
    """Qcodes driver for Zurich Instruments HF2LI lockin amplifier.

    This driver is meant to emulate a single-channel lockin amplifier,
    so one instance has a single demodulator, a single sigout channel,
    and multiple auxout channels (for X, Y, R, Theta, or an arbitrary manual value).
    Multiple instances can be run simultaneously as independent lockin amplifiers.

    This instrument has a great deal of additional functionality that is
    not currently supported by this driver.

    Args:
        name: Name of instrument.
        device: Device name, e.g. "dev204", used to create zhinst API session.
        demod: Index of the demodulator to use.
        sigout: Index of the sigout channel to use as excitation source.
        auxouts: Dict of the form {output: index},
            where output is a key of HF2LI.OUTPUT_MAPPING, for example {"X": 0, "Y": 3}
            to use the instrument as a lockin amplifier in X-Y mode with auxout channels 0 and 3.
        sigout2mixer: mapping from sigout to mixers. For default HF2LI {0:6, 1:7}
        num_sigout_mixer_channels: Number of mixer channels to enable on the sigouts. Default: 1.
    """
    OUTPUT_MAPPING = {-1: 'manual', 0: 'X', 1: 'Y', 2: 'R', 3: 'Theta'}
    def __init__(self, name: str, device: str, demod: int, sigout: int,
            auxouts: Dict[str, int], 
            sigout2mixer : Dict[ int, int ]={0:6, 1:7},
            num_sigout_mixer_channels: int=1, 
            **kwargs) -> None:

        super().__init__(name, **kwargs)
        instr = zhinst.utils.create_api_session(device, 1 )#, 
            #required_devtype='HF2LI') #initializes the instrument
        self.daq, self.dev_id, self.props = instr
        self.demod = demod
        self.sigout = sigout
        self.auxouts = auxouts
        log.info(f'Successfully connected to {name}.')
        self.sigout2mixer = sigout2mixer

        for ch in self.auxouts:
            self.add_parameter(
                name=ch,
                label=f'Scaled {ch} output value',
                unit='V',
                get_cmd=lambda channel=ch: self._get_output_value(channel),
                get_parser=float,
                docstring=f'Scaled and demodulated {ch} value.'
            )
            self.add_parameter(
                name=f'gain_{ch}',
                label=f'{ch} output gain',
                unit='V/Vrms',
                get_cmd=lambda channel=ch: self._get_gain(channel),
                get_parser=float,
                set_cmd=lambda gain, channel=ch: self._set_gain(gain, channel),
                vals=vals.Numbers(),
                docstring=f'Gain factor for {ch}.'
            )
            self.add_parameter(
                name=f'offset_{ch}',
                label=f'{ch} output offset',
                unit='V',
                get_cmd=lambda channel=ch: self._get_offset(channel),
                get_parser=float,
                set_cmd=lambda offset, channel=ch: self._set_offset(offset, channel),
                vals=vals.Numbers(-2560, 2560),
                docstring=f'Manual offset for {ch}, applied after scaling.'
            )
            self.add_parameter(
                name=f'output_{ch}',
                label=f'{ch} outptut select',
                get_cmd=lambda channel=ch: self._get_output_select(channel),
                get_parser=str
            )
            # Making output select only gettable, since we are
            # explicitly mapping auxouts to X, Y, R, Theta, etc.
            self._set_output_select(ch)
            
        self.add_parameter(
            name='ext_clk',
            label='External Clock',
            unit='',
            set_cmd=self._set_ext_clk,
            get_cmd=self._get_ext_clk,
            vals=vals.Bool()
        )
        
        self.add_parameter(
            name='phase',
            label='Phase',
            unit='deg',
            get_cmd=self._get_phase,
            get_parser=float,
            set_cmd=self._set_phase,
            vals=vals.Numbers(-180,180)
        )
        # self.add_parameter( #duplicate with timeconstant?
        #     name='time_constant',
        #     label='Time constant',
        #     unit='s',
        #     get_cmd=self._get_time_constant,
        #     get_parser=float,
        #     set_cmd=self._set_time_constant,
        #     vals=vals.Numbers()
        # )  
        self.osc=0 # set_frequency will set this oscillator
        self.add_parameter(
            name='frequency',
            label='Frequency',
            unit='Hz',
            get_cmd=self._get_frequency,
            set_cmd=self._set_frequency,
            get_parser=float
        ) 
        self.add_parameter(
            name='sigout_range',
            label='Signal output range',
            unit='V',
            get_cmd=self._get_sigout_range,
            get_parser=float,
            set_cmd=self._set_sigout_range,
            vals=vals.Enum(0.01, 0.1, 1, 10)
        )
         
        self.add_parameter(
            name='sigout_offset',
            label='Signal output offset',
            unit='V',
            snapshot_value=True,
            set_cmd=self._set_sigout_offset,
            get_cmd=self._get_sigout_offset,
            vals=vals.Numbers(-1, 1),
            docstring='Multiply by sigout_range to get actual offset voltage.'
        )

        self.add_parameter(
            name='sigout_dc_offset',
            label='Signal output DC offset',
            unit='V',
            snapshot_value=True,
            set_cmd=self._set_dc_offset,
            get_cmd=self._get_dc_offset,
            vals=vals.Numbers(-10, 10),
            docstring='Multiply by sigout_range to get actual offset voltage.'
        )

        single_values = (('x', 'Demodulated x', 'V'),
                    ('y', 'Demodulated y', 'V') )

        for name, unit, label in single_values :
            self.add_parameter( f'demod_{name}',
                    unit=unit,
                    label=label,
                    get_cmd = partial( self._single_get, name )
            )

        self.add_parameter(
                name=f'demod_theta',
                label=f'Demodulated theta'+ str(self.demod),
                unit='deg',
                get_cmd= self._get_theta,
                get_parser=float
            )
        
        demod_params = ( 
            ('timeconstant', 's'),
            ('order', ''),
            ('rate', '')
        )
        for param, unit in demod_params :
            self.add_parameter(
                    name=param,
                    label=param,
                    unit=unit,
                    get_cmd= partial( self._get_demod_param, param ),
                    set_cmd= partial( self._set_demod_param, param ),
                    get_parser=float
                )
        
        #### Parameters for Spectrum

        self.daq.sync()
        zoomfft = self.daq.zoomFFT()
        zoomfft.set("device", self.dev_id)
        self.daq_module = self.daq.dataAcquisitionModule()
        self.zoomfft = zoomfft
        #### Parameters for sweeping
        sweeper = self.daq.sweep()
        sweeper.set("device", self.dev_id)
        ### different ones?!
        sweeper.set("gridnode", f"oscs/{self.sigout}/freq") ### Set the sweeping parameter
        self.sweeper = sweeper
        # do an initial trigger for snapshot

        sweeper_params = ( ( 'samplecount', '', 'Points' ),
                ( 'start', 'Hz', 'Start frequency' ),
                ('stop', 'Hz', 'Stop frequency' ),
                ('xmapping', '', 'X scale as log or linear'),
                ('bandwidthoverlap', '', 'Bandwidth Overlap') )

        for namex, unit, label in sweeper_params :
            self.add_parameter( f'sweeper_{namex}',
                    unit=unit,
                    label=label,
                    set_cmd = partial( self.sweeper.set, namex ),
                    get_cmd = partial( self._sweeper_get, namex )
            )

        self.add_parameter( 'trace_frequency',
                unit='Hz',
                label= 'Frequency',
                snapshot_value=False,
                get_cmd= lambda : self.samples['grid'],
                vals=vals.Arrays(shape=(self.sweeper_samplecount,))
            )

        self._averages=1
        self.add_parameter( 'averages',
                unit='npts',
                label= 'Averaging',
                set_cmd = partial( setattr, self, '_averages' ),
                get_cmd = partial( getattr, self, '_averages' )
            )  
  
        self.auto_trigger = False 

        for p, units in ( ('r', 'dB'), ('x','dB'), ('y','dB'),('phase', 'deg') ) :
            self.add_parameter( f'trace_{p}',
                    unit= units,
                    label= p,
                    parameter_class = ParameterWithSetpoints,
                    setpoints = ( self.trace_frequency,),
                    get_cmd= partial(self._get_sweep_param, p ),
                    vals=vals.Arrays(shape=(self.sweeper_samplecount,))
                )

        self.add_parameter( 'spectrum_frequency',
                unit='Hz',
                label= 'Frequency',
                snapshot_value=False,
                get_cmd= lambda : self.spectrum_samples[0][0]["grid"],
                vals=vals.Arrays(shape=(self._spectrum_freq_length,))
            )

        # each psd type must have an associated _process_psd_xx function
        # see self._get_spectrum for details
        for p, units in ( ('psd_corrected', '$V^2/Hz$'), 
            ('psd', '$V^2/Hz$'), ('psd_i', '$V^2/Hz$'),
            ('psd_q', '$V^2/Hz$'),('psd_iq', '$V^2/Hz$'),
            ('psd_xx', '$V^2/Hz$'),('psd_yy', '$V^2/Hz$'),
            ('psd_xy', '$V^2/Hz$')
            ):
            self.add_parameter( p,
                    unit= units,
                    label= p,
                    parameter_class = ParameterWithSetpoints,
                    setpoints = ( self.spectrum_frequency,),
                    get_cmd= partial(self._get_spectrum, p ),
                    vals=vals.Arrays(shape=(self._spectrum_freq_length,))
            )


        self._bits = 8
        self.add_parameter( 'psd_points', 
            unit = '',
            label = 'Points',
            set_cmd = self._set_points,
            get_cmd = lambda : 2**self._bits
        )

        for output, mixer_channel in sigout2mixer.items():
        # for i in range(6, num_sigout_mixer_channels):
            self.add_parameter(
                name=f'sigout_enable{mixer_channel}',
                label=f'Signal output mixer {mixer_channel} enable',
                get_cmd=lambda : self._get_sigout_enable(mixer_channel, output),
                get_parser=float,
                set_cmd=lambda amp : self._set_sigout_enable(mixer_channel, output, amp),
                vals=vals.Enum(0,1,2,3),
                docstring="""\
                0: Channel off (unconditionally)
                1: Channel on (unconditionally)
                2: Channel off (will be turned off on next change of sign from negative to positive)
                3: Channel on (will be turned on on next change of sign from negative to positive)
                """
            )
            self.add_parameter(
                name=f'sigout_amplitude{mixer_channel}',
                label=f'Signal output mixer {mixer_channel} amplitude',
                unit='Gain',
                get_cmd=partial( self._get_sigout_amplitude, mixer_channel, output ),
                get_parser=float,
                set_cmd=partial( self._set_sigout_amplitude, mixer_channel, output ),
                vals=vals.Numbers(-10, 10),
                docstring='Multiply by sigout_range to get actual output voltage.'
            )

    def _spectrum_freq_length(self):
        #return self.daq_module.get('grid/cols')['grid']['cols'][0]-1
        return len(self.spectrum_samples[0][0]["grid"])

    def _get_time_constant(self):
        path = f'/{self.dev_id}/demods/{self.demod}/timeconstant/'
        return self.daq.getDouble(path)
    def _set_time_constant(self, timeconstant):
        path = f'/{self.dev_id}/demods/{self.demod}/timeconstant/'
        return self.daq.setDouble(path, timeconstant)

    def _sweeper_get( self, name ) :
        """ wrap zi sweeper.get
        """
        return self.sweeper.get( name )[name][0]

    def _single_get(self, name):
        path = f'/{self.dev_id}/demods/{self.demod}/sample/'
        return self.daq.getSample(path)[name][0]
    
    def _set_ext_clk(self, val):
        """ set external 10 MHz clock
        """
        path = f'/{self.dev_id}/system/extclk'
        self.daq.setInt(path, int(val) )

    def _get_ext_clk( self ):
        """ get external 10 MHz clock as bool
        """
        path = f'/{self.dev_id}/system/extclk'
        val = self.daq.getInt( path )
        return bool( val )

    def _get_sweep_param(self, param, fr=True):
        if self.auto_trigger :
            self.trigger_sweep()

        if param == 'phase' :
            values = (self.samples[param])*180/np.pi
        else :
            # detect which node we are sweeping with
            osc = self.osc
            mixer = self.sigout2mixer[osc]
            amplitude = self._get_sigout_amplitude( mixer, osc ) / ( 2 * np.sqrt(2) ) # normalization factor for vpp 2x fudge
            values = 20 * np.log10( self.samples[param]/amplitude )

        return values

    def _get_spectrum(self, param ):
        """ return spectrum in units of V**2/Hz
        """

        if self.auto_trigger :
            self.trigger_spectrum()

        processor = getattr( self, f'_process_{param}' )
        data = processor()

        data = np.mean( data, axis=0 )
        bw = self.rate() / self.psd_points()
        data = data / bw
        # return values
        return data

    def _process_psd_corrected( self ) :
        """ perform processing for corrected psd
        returns data, ready for averaging
        """

        xiy = lambda entry : entry[0]['x'] + 1j * entry[0]['y']
        data = [ xiy( entry ) for entry in self.spectrum_samples ]


        filter = self.spectrum_samples[0][0]['filter']
        data = [ entry / filter for entry in data]

        data = np.array( data )
        return np.abs( data )**2

    def _process_psd( self ) :
        """ perform processing for psd
        """
        xiy = lambda entry : entry[0]['x'] + 1j * entry[0]['y']
        data = [ xiy( entry ) for entry in self.spectrum_samples ]
        data = np.array( data )
        return np.abs( data )**2
    
    def _normalize_spectra(self, data) :
        """ normalize spectrum to filter.
        Args:
            data: list of data to normalize
        Returns:
            normalized data as np array
        """
        filter = self.spectrum_samples[0][0]['filter']
        return np.array( [ entry / filter for entry in data] )

    def _process_psd_xx(self) :
        """ x psd """
        x = lambda entry : entry[0]['x']
        data = [ x( entry ) for entry in self.spectrum_samples ]
        data = self._normalize_spectra( data )
        return data**2

    def _process_psd_yy(self) :
        """ y psd """
        y = lambda entry : entry[0]['y']
        data = [ y( entry ) for entry in self.spectrum_samples ]
        data = self._normalize_spectra( data )
        return data**2

    def _process_psd_xy(self) :
        """ xy psd """
        x = lambda entry : entry[0]['x']
        xdata = [ x( entry ) for entry in self.spectrum_samples ]
        xdata = self._normalize_spectra( xdata )

        y = lambda entry : entry[0]['y']
        ydata = [ y( entry ) for entry in self.spectrum_samples ]
        ydata = self._normalize_spectra( ydata )
        return xdata*ydata

    def _process_psd_i(self) :

        def i( entry ) :
            x = entry[0]['x']
            y = entry[0]['y']
            i = (x + x[::-1] )/2 + 1j * (y - y[::-1])/2
            return i

        data = [ i( entry ) for entry in self.spectrum_samples ]

        data = self._normalize_spectra( data )

        return np.abs( data )**2

    def _process_psd_q(self) :
        def q( entry ) :
            x = entry[0]['x']
            y = entry[0]['y']
            q = (x - x[::-1] )/2j + 1j * (y + y[::-1])/2j
            return q

        data = [ q( entry ) for entry in self.spectrum_samples ]
    
        data = self._normalize_spectra( data )
        return np.abs( data )**2

    def _process_psd_iq(self) :
        xiyQ = lambda entry : (entry[0]['x']-entry[0]['x'][::-1] + 1j * (entry[0]['y']+entry[0]['y'][::-1])/(2*1j))
        dataQ = [ xiyQ( entry ) for entry in self.spectrum_samples ]
        
        xiyI = lambda entry : (entry[0]['x']+entry[0]['x'][::-1] + 1j * (entry[0]['y']-entry[0]['y'][::-1])/2)
        dataI = [ xiyI( entry ) for entry in self.spectrum_samples ]
        
        dataI = self._normalize_spectra( dataI )
        dataQ = self._normalize_spectra( dataQ )

        dataIQ = dataI*np.conjugate(dataQ)

        return np.real( dataIQ )

    def _get_theta(self):
        path = f'/{self.dev_id}/demods/{self.demod}/sample/'
        sample = self.daq.getSample(path)
        cmplx = sample['x'] + 1j*sample['y']
        return np.angle(cmplx)*180/np.pi

    def bw3db( self ) :
        """ Return 3dB bandwidth of self.demod
        """
        zi = self
        zi.order(2)
        o = zi.order()
        tc = zi.timeconstant()
        return np.sqrt(2**(1/(o))-1)/tc / ( 2 * np.pi )

    def trigger_sweep(self):
        sweeper = self.daq.sweep()
        #self.snapshot(update=True)
        #sweeper = self.sweeper
        sweeper.set("device", self.dev_id)
        sweeper.set('gridnode', f'oscs/{self.osc}/freq')
        sweeper.set('scan', 0) ### Sequenctial sweep
        sweeper.set("bandwidthcontrol", 0) ### Bandwidth control: Auto
        #sweeper.set('maxbandwidth', 100) ### Max demodulation bandwidth
        sweeper.set('settling/inaccuracy', 1.0e-08)
        path = f"/{self.dev_id}/demods/{self.demod}/sample"
        sweeper.set("start", self.sweeper_start())
        sweeper.set("stop", self.sweeper_stop())
        sweeper.set("samplecount", self.sweeper_samplecount()) 
        #sweeper.set()
        self.timeconstant(self.timeconstant())
        sweeper.subscribe(path)
        sweeper.execute()

        ### Wait until measurement is done 
        start_t = time.time()
        timeout = 6000  # [s]
        while not sweeper.finished():  # Wait until the sweep is complete, with timeout.
            time.sleep(1)
            progress = sweeper.progress()
            if (time.time() - start_t) > timeout:
                print("\nSweep still not finished, forcing finish...")
                sweeper.finish()
        data = sweeper.read(True)
        self.blob = data
        self.samples = data[path][0][0]
        sweeper.unsubscribe(path) ### Unsubscribe from the signal path

    def trigger_spectrum(self):
            zoomfft = self.zoomfft
            #self.snapshot(update=True)
            zoomfft.set("mode", 0)
            zoomfft.set("overlap", 0)
            # 0=Rectangular, 1=Hann, 2=Hamming, 3=Blackman Harris,
            # 16=Exponential, 17=Cosine, 18=Cosine squared.
            zoomfft.set("window", 1)
            zoomfft.set("absolute", 1) # Return absolute frequencies instead of relative to 0.
            zoomfft.set("bit", self._bits ) # The number of lines is 2**bits.
            zoomfft.set("loopcount", self.averages() )
            self.daq_module.set('spectrum/autobandwidth', 1) # automatically set bandwidth
            # self.daq_module.set('grid/repetitions', 50)
            path = "/%s/demods/%d/sample" % (self.dev_id, self.demod)
            zoomfft.subscribe(path)
            zoomfft.execute()

            start = time.time()
            timeout = 60000  # [s]

            while not zoomfft.finished():
                time.sleep(0.2)
                progress = zoomfft.progress()
                if (time.time() - start) > timeout:
                    print("\nzoomFFT still not finished, forcing finish...")
                    zoomfft.finish()
            print("")

            return_flat_data_dict = True
            data = zoomfft.read(return_flat_data_dict)
            self.spectrum_samples = data[path]
            zoomfft.unsubscribe(path)
            

    def _get_data(self, poll_length=0.1) -> float:
        path = f'/{self.dev_id}/demods/{self.demod}/sample'
        self.daq.unsubscribe("*")
        poll_timeout = 500  # [ms]
        poll_flags = 0
        poll_return_flat_dict = True
        self.daq.sync()
        self.daq.subscribe(path)
        data = self.daq.poll(poll_length, poll_timeout, poll_flags, poll_return_flat_dict)
        self.daq.unsubscribe("*")
        return data

    def readout(self, poll_length : Optional[float] = 0.1 ):
        """ record self.demod
        Args:
            poll_length: length of time in seconds to record for
        Returns:
            X, Y, t as np arrays
        """
        path = f'/{self.dev_id}/demods/{self.demod}/sample'
        data = self._get_data( poll_length=poll_length )
        sample = data[path]
        X = sample['x']
        Y = sample['y']
        clockbase = float(self.daq.getInt(f'/{self.dev_id}/clockbase'))
        t = (sample['timestamp'] - sample['timestamp'][0]) / clockbase 
        return (X, Y, t)

    def _set_points( self, points ) :
        """ set number of fft points to the nearest power of 2
        """
        self._bits = np.round( np.log2( points ) )

    def _get_phase(self) -> float:
        path = f'/{self.dev_id}/demods/{self.demod}/phaseshift/'
        return self.daq.getDouble(path)

    def _set_phase(self, phase: float) -> None:
        path = f'/{self.dev_id}/demods/{self.demod}/phaseshift/'
        self.daq.setDouble(path, phase)
        
    def _get_gain(self, channel: str) -> float:
        path = f'/{self.dev_id}/auxouts/{self.auxouts[channel]}/scale/'
        return self.daq.getDouble(path)

    def _set_gain(self, gain: float, channel: str) -> None:
        path = f'/{self.dev_id}/auxouts/{self.auxouts[channel]}/scale/'
        self.daq.setDouble(path, gain)

    def _get_offset(self, channel: str) -> float:
        path = f'/{self.dev_id}/auxouts/{self.auxouts[channel]}/offset/'
        return self.daq.getDouble(path)

    def _set_offset(self, offset: float, channel: str) -> None:
        path = f'/{self.dev_id}/auxouts/{self.auxouts[channel]}/offset/'
        self.daq.setDouble(path, offset)

    def _get_output_value(self, channel: str) -> float:
        path = f'/{self.dev_id}/auxouts/{self.auxouts[channel]}/value/'
        return self.daq.getDouble(path)

    def _get_output_select(self, channel: str) -> str:
        path = f'/{self.dev_id}/auxouts/{self.auxouts[channel]}/outputselect/'
        idx = self.daq.getInt(path)
        return self.OUTPUT_MAPPING[idx]

    def _set_output_select(self, channel: str) -> None:
        path = f'/{self.dev_id}/auxouts/{self.auxouts[channel]}/outputselect/'
        keys = list(self.OUTPUT_MAPPING.keys())
        idx = keys[list(self.OUTPUT_MAPPING.values()).index(channel)]
        self.daq.setInt(path, idx)

    def _get_demod_param( self, param ) :
        """ get demod parameter
        Args:
            param: string parameter name. eg timeconstant\
        Returns:
            parameter value as a double
        """
        path = f'/{self.dev_id}/demods/{self.demod}/{param}/'
        return self.daq.getDouble(path)


    def _set_demod_param( self, param, value ) :
        """ set demod parameter
        Args:
            param: string parameter name. eg timeconstant\
        Returns:
            parameter value as a double
        """
        path = f'/{self.dev_id}/demods/{self.demod}/{param}/'
        self.daq.setDouble(path, value)

    def _get_time_constant(self) -> float:
        path = f'/{self.dev_id}/demods/{self.demod}/timeconstant/'
        return self.daq.getDouble(path)

    def _set_time_constant(self, tc: float) -> None:
        path = f'/{self.dev_id}/demods/{self.demod}/timeconstant/'
        self.daq.setDouble(path, tc)

    def _get_sigout_range(self, sigout=None ) -> float:
        if sigout is None :
            sigout = self.sigout
        path = f'/{self.dev_id}/sigouts/{sigout}/range/'
        return self.daq.getDouble(path)

    def _set_sigout_range(self, rng: float, sigout=None ) -> None:
        if sigout is None :
            sigout = self.sigout       
        path = f'/{self.dev_id}/sigouts/{sigout}/range/'
        self.daq.setDouble(path, rng)
    
    def _set_dc_range(self, rng: float) -> None:
        path = f'/dev1792/sigouts/1/range/'
        self.daq.setDouble(path, rng)

    def _get_dc_range(self) -> float:
        path = f'/dev1792/sigouts/1/range/'
        return self.daq.getDouble(path)
    
    def _get_dc_offset(self) -> float:
        path = f'/dev1792/sigouts/1/offset/'
        range = self._get_dc_range()
        return self.daq.getDouble(path)*range

    def _set_dc_offset(self, offset: float) -> None:
        path = f'/dev1792/sigouts/1/offset/'
        range = self._get_dc_range()
        return self.daq.setDouble(path, offset/range)

    def _get_sigout_offset(self) -> float:
        path = f'/{self.dev_id}/sigouts/{self.sigout}/offset/'
        range = self._get_sigout_range()
        return self.daq.getDouble(path)*range

    def _set_sigout_offset(self, offset: float) -> None:
        path = f'/{self.dev_id}/sigouts/{self.sigout}/offset/'
        range = self._get_sigout_range()
        return self.daq.setDouble(path, offset/range)

    def _get_sigout_amplitude(self, mixer_channel: int, sigout:int ) -> float:
        path = f'/{self.dev_id}/sigouts/{sigout}/amplitudes/{mixer_channel}/'
        range = self._get_sigout_range(sigout=sigout)
        return self.daq.getDouble(path)*range

    def _set_sigout_amplitude(self, mixer_channel: int, sigout:int, amp: float) -> None:
        path = f'/{self.dev_id}/sigouts/{sigout}/amplitudes/{mixer_channel}/'
        range = self._get_sigout_range(sigout=sigout)
        return self.daq.setDouble(path, amp/range)

    def _get_sigout_enable(self, mixer_channel: int, sigout: int ) -> int:
        path = f'/{self.dev_id}/sigouts/{sigout}/enables/{mixer_channel}/'
        return self.daq.getInt(path)

    def _set_sigout_enable(self, mixer_channel: int, sigout: int, val: int) -> None:
        path = f'/{self.dev_id}/sigouts/{sigout}/enables/{mixer_channel}/'
        self.daq.setInt(path, val)

    def _get_frequency(self) -> float:
        path = f'/{self.dev_id}/demods/{self.demod}/freq/'
        return self.daq.getDouble(path)

    def _set_frequency(self, freq) -> float:
        osc_index = self.osc
        return self.daq.set([["/%s/oscs/%d/freq" % (self.dev_id, osc_index), freq]])

    def sample(self) -> dict:
        path = f'/{self.dev_id}/demods/{self.demod}/sample/'
        return self.daq.getSample(path)
        
    def ask(self,arg) :
        """" hacking in an ask method
        """
        if arg == "*IDN?" :
            return self.dev_id
        else :
            raise ValueError(f"I don't understand {arg}")