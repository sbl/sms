#!/usr/bin/env python

import click


def subtract(filename, params):
    '''
    Subtract the sinusoidal components as computed by the sinusoidal model.
    '''
    import os.path
    import essentia
    from essentia.streaming import (MonoLoader, MonoWriter, FrameCutter,
                                    Windowing, SineModelAnal, SineSubtraction,
                                    FFT, VectorInput)

    outfile = os.path.join(
        os.path.dirname(filename),
        os.path.splitext(os.path.basename(filename))[0] + '-smssubstract.wav')

    loader = MonoLoader(filename=filename, sampleRate=params['sampleRate'])
    pool = essentia.Pool()
    fcut = FrameCutter(
        frameSize=params['frameSize'], hopSize=params['hopSize'])
    w = Windowing(type="blackmanharris92")
    fft = FFT(size=params['frameSize'])
    smanal = SineModelAnal(
        sampleRate=params['sampleRate'],
        maxnSines=params['maxnSines'],
        magnitudeThreshold=params['magnitudeThreshold'],
        freqDevOffset=params['freqDevOffset'],
        freqDevSlope=params['freqDevSlope'])
    subtrFFTSize = min(params['frameSize'] / 4, 4 * params['hopSize'])
    smsub = SineSubtraction(
        sampleRate=params['sampleRate'],
        fftSize=subtrFFTSize,
        hopSize=params['hopSize'])

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

    # store to file
    outaudio = pool['frames'].flatten()

    awrite = MonoWriter(filename=outfile, sampleRate=params['sampleRate'])
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
def cli(filename, framesize, hopsize, samplerate, maxnsines):
    params = {
        'frameSize': framesize,
        'hopSize': hopsize,
        'sampleRate': samplerate,
        'maxnSines': maxnsines,
        'magnitudeThreshold': -74,
        'minSineDur': 0.02,
        'freqDevOffset': 10,
        'freqDevSlope': 0.001
    }
    subtract(filename, params)


if __name__ == '__main__':
    cli()
