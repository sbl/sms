import os.path
import click

from subprocess import call


def subtract(filename, outfile, params):
    '''
    Subtract the sinusoidal components as computed by the sinusoidal model.
    '''
    import essentia
    from essentia.streaming import (MonoLoader, MonoWriter, FrameCutter,
                                    Windowing, SineModelAnal, SineSubtraction,
                                    FFT, VectorInput)

    loader = MonoLoader(filename=filename, sampleRate=params['sampleRate'])
    awrite = MonoWriter(filename=outfile, sampleRate=params['sampleRate'])

    fcut = FrameCutter(
        frameSize=params['frameSize'], hopSize=params['hopSize'])

    w = Windowing(type="blackmanharris92")
    fft = FFT(size=params['frameSize'])

    smanal = SineModelAnal(
        sampleRate=params['sampleRate'],
        maxnSines=params['maxnSines'],
        magnitudeThreshold=params['magnitudeThreshold'],
        freqDevOffset=params['freqDevOffset'],
        freqDevSlope=params['freqDevSlope'],
        minFrequency=params['minFrequency'],
        maxFrequency=params['maxFrequency'], )

    subtrFFTSize = min(params['frameSize'] / 4, 4 * params['hopSize'])
    smsub = SineSubtraction(
        sampleRate=params['sampleRate'],
        fftSize=subtrFFTSize,
        hopSize=params['hopSize'])

    pool = essentia.Pool()

    # analysis
    loader.audio >> fcut.signal
    fcut.frame >> w.frame
    w.frame >> fft.frame
    fft.fft >> smanal.fft
    # subtraction
    fcut.frame >> smsub.frame
    smanal.magnitudes >> smsub.magnitudes
    smanal.frequencies >> smsub.frequencies
    smanal.phases >> smsub.phases
    smsub.frame >> (pool, 'frames')

    essentia.run(loader)

    outaudio = pool['frames'].flatten()
    outvector = VectorInput(outaudio)

    outvector.data >> awrite.audio
    essentia.run(outvector)


@click.command()
@click.argument('filename', type=click.Path(True))
@click.option(
    '--framesize', default=2048, help='framesize power of 2', type=int)
@click.option('--hopsize', default=128, help='hopsize power of 2', type=int)
@click.option('--samplerate', default=44100, help='samplerate', type=int)
@click.option(
    '--maxnsines', default=100, help='maximum number of sines', type=int)
@click.option(
    '--thresh',
    default=-74,
    help='peaks below this given threshold are not outputted',
    type=int)
@click.option(
    '--freqdevslope',
    default=0.01,
    help='peaks below this given threshold are not outputted',
    type=float)
@click.option(
    '--freqdevoffset',
    default=20,
    help='minimum frequency deviation at 0Hz',
    type=int)
@click.option(
    '--minfreq',
    default=20,
    help='the minimum frequency of the range to evaluate [Hz]',
    type=int)
@click.option(
    '--maxfreq',
    default=22050,
    help='the maximum frequency of the range to evaluate [Hz]',
    type=int)
def cli(filename, framesize, hopsize, samplerate, maxnsines, minfreq, maxfreq,
        thresh, freqdevslope, freqdevoffset):
    params = {
        'frameSize': framesize,
        'hopSize': hopsize,
        'sampleRate': samplerate,
        'maxnSines': maxnsines,
        'minFrequency': minfreq,
        'maxFrequency': maxfreq,
        'magnitudeThreshold': thresh,
        'freqDevOffset': freqdevoffset,
        'freqDevSlope': freqdevslope,
    }
    outfile = os.path.join(
        os.path.dirname('.'), 'snd',
        os.path.splitext(os.path.basename(filename))[0] + '-smssubstract.wav')

    subtract(filename, outfile, params)
    call(['afplay', outfile])


if __name__ == '__main__':
    cli()
