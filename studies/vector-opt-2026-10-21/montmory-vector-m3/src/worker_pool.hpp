#ifndef MONTMORY_WORKER_POOL_HPP
#define MONTMORY_WORKER_POOL_HPP
#include "stream.hpp"
#include <condition_variable>
#include <mutex>
#include <thread>
#include <vector>
namespace montmory_stream {
// Synchronous reusable pool; one caller at a time, enforced by submit_mutex.
// No affinity or physical core ownership is assumed.
class WorkerPool {
    std::mutex mutex,submit_mutex;
    std::condition_variable work,complete;
    std::vector<std::thread> threads;
    bool stopping=false;
    uint64_t epoch=0;
    unsigned remaining=0;
    uint64_t *output=nullptr,count=0,seed=0,base=0;
    Engine engine=Engine::pair;
    unsigned shift=0;
    void loop(unsigned id,unsigned workers) {
        uint64_t seen=0;
        std::unique_lock<std::mutex> lock(mutex);
        for(;;) {
            work.wait(lock,[&]{return stopping || epoch!=seen;});
            if(stopping)return;
            seen=epoch;
            // Split without multiplication overflow. Empty partitions do nothing.
            uint64_t q=count/workers,r=count%workers;
            uint64_t begin=q*id+std::min<uint64_t>(id,r);
            uint64_t size=q+(id<r);
            auto* out=output;auto s=seed,b=base;auto e=engine;auto sh=shift;
            lock.unlock();
            if(size)fill(out+begin,size,s,b+begin,e,sh);
            lock.lock();
            if(--remaining==0)complete.notify_one();
        }
    }
public:
    explicit WorkerPool(unsigned workers) {
        if(workers==0)throw std::invalid_argument("Zero workers");
        threads.reserve(workers);
        try{for(unsigned i=0;i<workers;++i)threads.emplace_back(&WorkerPool::loop,this,i,workers);}
        catch(...) {
            {std::lock_guard<std::mutex> lock(mutex);stopping=true;}
            work.notify_all();for(auto& t:threads)t.join();throw;
        }
    }
    WorkerPool(const WorkerPool&)=delete;
    WorkerPool& operator=(const WorkerPool&)=delete;
    ~WorkerPool() {
        {std::lock_guard<std::mutex> lock(mutex);stopping=true;}
        work.notify_all();for(auto& t:threads)t.join();
    }
    void generate(uint64_t* out,uint64_t n,uint64_t s,uint64_t b,Engine e=Engine::pair,unsigned sh=0) {
        if((n&&!out)||(n&&n-1>UINT64_MAX-b)||(sh!=0 && sh!=11) || unsigned(e)>unsigned(Engine::simd8))
            throw std::invalid_argument("Invalid pool request");
        std::lock_guard<std::mutex> caller(submit_mutex);
        std::unique_lock<std::mutex> lock(mutex);
        output=out;count=n;seed=s;base=b;engine=e;shift=sh;
        remaining=unsigned(threads.size());++epoch;work.notify_all();
        complete.wait(lock,[&]{return remaining==0;});
    }
};
}
#endif
