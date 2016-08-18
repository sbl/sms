import click
import os

from subprocess import call


def stoch(filename, outfile, params):
    import essentia
    import essentia.streaming as es

    loader = es.EasyLoader(filename=filename)
    fcut = es.FrameCutter(
        frameSize=params['frameSize'], hopSize=params['hopSize'])

    stochAnal = es.StochasticModelAnal(
        fftSize=params['frameSize'],
        hopSize=params['hopSize'],
        stocf=params['stocf'])

    stochSynth = es.StochasticModelSynth(
        fftSize=params['frameSize'],
        hopSize=params['hopSize'],
        stocf=params['stocf'])

    pool = essentia.Pool()

    loader.audio >> fcut.signal
    fcut.frame >> stochAnal.frame
    stochAnal.stocenv >> stochSynth.stocenv

    stochSynth.frame >> (pool, 'frames')

    # do the processing
    essentia.run(loader)

    outaudio = pool['frames'].flatten()
    outvector = es.VectorInput(outaudio)

    awrite = es.MonoWriter(filename=outfile)
    outvector.data >> awrite.audio

    essentia.run(outvector)


@click.command()
@click.argument('filename', type=click.Path(True))
@click.option(
    '--framesize', default=2048, help='framesize power of 2', type=int)
@click.option('--hopsize', default=512, help='hopsize', type=int)
@click.option('--stocf', default=0.1, help='stochastic factor', type=float)
def cli(filename, framesize, hopsize, stocf):
    params = {'frameSize': framesize, 'hopSize': hopsize, 'stocf': stocf}

    outfile = os.path.join(
        os.path.dirname(filename),
        os.path.splitext(os.path.basename(filename))[0] + '-stoch.wav')

    # process
    stoch(filename, outfile, params)

    # listen
    call(['afplay', outfile])


if __name__ == '__main__':
    cli()
