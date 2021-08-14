from multiprocessing import Process, Pool
import time

def func1(data):
    
    time.sleep(2)
    return data




if __name__ == '__main__':
    numProcess = 20
    numTask = 50
    pool = Pool(numProcess)
    result = []
    for i in range(numTask):
        result.append(pool.apply_async(func=func1, args=(i,)))

    pool.close()
    pool.join()
    print('done')

    for res in result:
        print(res.get())