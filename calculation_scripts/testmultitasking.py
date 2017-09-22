# -*- coding=utf-8 -*-
def test():
    print('Hello, world!')

def printtest(N):
    import multiprocessing as mp




    queuelist = [mp.Queue() for i in xrange(N)]
    processlist = [mp.Process(target = test, args = ()) for i in xrange(N)]
    for process in processlist:
        process.start()
    for process in processlist:
        process.join()



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=('Kake?'))
    parser.add_argument('-N', type=int, default=4, help='The number of processes')
    args = parser.parse_args()
    printtest(N=args.N)
