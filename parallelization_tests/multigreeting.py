def writestuff(procno, totproc):
    print('Hello, world! From process {} of {}'.format(procno, totproc))

def multirun(N):
    import multiprocessing as mp

    processlist = [mp.Process(target = writestuff, args = (i, N)) for i in range(N)]
    for process in processlist:
        process.start()
    for process in processlist:
        process.join()





if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=('Write a greeting any no.'
                                                  'of times'))
    parser.add_argument('-N', type = int, default = 1,
                        help = 'The number of greetings you want to receive')
    args = parser.parse_args()

    multirun(N=args.N)

