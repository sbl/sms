import os.path
import click

from subprocess import call


def sines(filename, outfile, params):
    '''
    Extract the sinusoidal components of an audio file
    '''
    import essentia
    from essentia.streaming import (MonoLoader, MonoWriter, FrameCutter,
                                    Windowing, SineModelAnal, SineModelSynth,
                                    FFT, IFFT, OverlapAdd)

    loader = MonoLoader(filename=filename, sampleRate=params['sampleRate'])
    awrite = MonoWriter(filename=outfile, sampleRate=params['sampleRate'])

    fcut = FrameCutter(
        frameSize=params['frameSize'], hopSize=params['hopSize'],
        startFromZero=False)
    overl = OverlapAdd(
        frameSize=params['frameSize'],
        hopSize=params['hopSize'],
        gain=1.0 / params['frameSize'])

    w = Windowing(type="blackmanharris92")
    fft = FFT(size=params['frameSize'])
    ifft = IFFT(size=params['frameSize'])

    smanal = SineModelAnal(
        sampleRate=params['sampleRate'],
        maxnSines=params['maxnSines'],
        magnitudeThreshold=params['magnitudeThreshold'],
        freqDevOffset=params['freqDevOffset'],
        freqDevSlope=params['freqDevSlope'],
        minFrequency=params['minFrequency'],
        maxFrequency=params['maxFrequency'], )

    syn = SineModelSynth(
        sampleRate=params['sampleRate'],
        fftSize=params['frameSize'],
        hopSize=params['hopSize'])

    # analysis
    loader.audio >> fcut.signal
    fcut.frame >> w.frame
    w.frame >> fft.frame
    fft.fft >> smanal.fft

    # synth
    smanal.magnitudes >> syn.magnitudes
    smanal.frequencies >> syn.frequencies
    smanal.phases >> syn.phases

    # ifft
    syn.fft >> ifft.fft
    ifft.frame >> overl.frame
    overl.signal >> awrite.audio

    essentia.run(loader)


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
        os.path.splitext(os.path.basename(filename))[0] + '-smssines.wav')

    sines(filename, outfile, params)
    call(['afplay', outfile])


if __name__ == '__main__':
    cli()
