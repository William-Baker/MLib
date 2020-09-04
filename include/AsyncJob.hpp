#include <queue>
#include <thread>
#include <condition_variable>
#include <stdio.h>


template<class Job> class AsyncJob{
    public:

    template<class sJob> class JobQueue{
        public:
            std::mutex mutex;
            std::queue<sJob*> jobs;
    };


    std::condition_variable condVar;
    std::thread* thread;
    JobQueue<Job> queue;
    bool doneWaiting = false;
    bool closeThread = false;
    std::mutex m_mutex;
    void* userData = 0;

    int (*callback)(Job*, void*);

    AsyncJob(int (*callback)(Job*, void*), void* userData){
        this->callback = callback;
        this->userData = userData;
        thread = new std::thread(AsyncJob::loop, this);

    }

    void addJob(Job* j){
        queue.mutex.lock();
        queue.jobs.push(j);
        queue.mutex.unlock();
        doneWaiting = true;
        condVar.notify_one();
    }

    static void loop(AsyncJob* aj){
        while(true){
            aj->queue.mutex.lock();
            if(aj->queue.jobs.size() == 0){
                aj->queue.mutex.unlock();
                std::unique_lock<std::mutex> mlock(aj->m_mutex);
                aj->condVar.wait(mlock, [aj]{return aj->doneWaiting;});
                aj->doneWaiting = false;
                aj->m_mutex.unlock();


                //check if we should close the thread
                if(aj->closeThread){
                    aj->closeThread = false;
                    std::cout << "thread terminating ";
                    return;
                }
            }

            aj->queue.mutex.try_lock();
            
            if(aj->queue.jobs.size() == 0) {
                aj->queue.mutex.unlock();
                continue; 
            }
            Job* job = aj->queue.jobs.front();
            aj->queue.jobs.pop();
            aj->queue.mutex.unlock();

            int success = aj->callback(job, aj->userData);


            if(success){
                aj->queue.mutex.lock();
                aj->queue.jobs.push(job);
                aj->queue.mutex.lock();
            }
            else{
                free(job);
            }
        }

    }

    ~AsyncJob(){
        std::cout << (" Deconstructing write thread...");
        while(queue.jobs.size() != 0){}
        std::cout << (" write jobs finished...");
        closeThread = true;
        doneWaiting = true;//this allways must be called before notify as a back up
        condVar.notify_one();
        std::cout << (" write thread notified...");
        while(closeThread){} //Thread will reset this once the notification has been recieved
        std::cout << (" write thread terminated ");
        //this->writeThread->~thread();
        std::cout << (" file deconstructed\n");
    }

};