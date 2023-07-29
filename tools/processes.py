from multiprocessing import Manager, Pool, cpu_count
from tqdm import tqdm
import csv


class ProcessHandler:
    @classmethod
    def handle(cls, func, param_setups, output_file: str, header):
        manager = Manager()
        queue = manager.Queue()
        pool = Pool(cpu_count() + 2)
        jobs = []

        pool.apply_async(cls.listener, (queue, output_file, header))

        for ps in param_setups:
            job = pool.apply_async(cls.worker, (queue, func, ps))
            jobs.append(job)

        for job in tqdm(jobs, desc="Multiprocessing method"):
            job.get()

        # now we are done, kill the listener
        queue.put('<KILL>')
        pool.close()
        pool.join()

    @classmethod
    def listener(cls, q, output_file, header: list):
        """Listen for messages from a queue. Then write them to csv."""

        with open(output_file, 'w', newline='\n', encoding='utf-8') as f:
            writer = csv.writer(f, delimiter=';')
            writer.writerow(header)
            f.flush()

            while 1:
                m = q.get()
                if m == '<KILL>':
                    print(f"Results writen into {output_file}.")
                    break

                writer.writerow(m.split(";"))
                f.flush()

    @classmethod
    def worker(cls, q, func, setup):
        res = func(setup)
        augmented_result = ';'.join(list(map(lambda x: str(x), list(setup.values()) + list(res.values))))
        q.put(augmented_result)


