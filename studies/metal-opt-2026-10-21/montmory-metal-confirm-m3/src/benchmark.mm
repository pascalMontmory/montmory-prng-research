#include "v3_helpers.hpp"
#include "lane_reference.hpp"
#include <cmath>

struct OptJob {unsigned engine,workers,group,shift;};
using JobKey=std::tuple<unsigned,unsigned,unsigned,unsigned>;
using Family=std::pair<unsigned,unsigned>;
static JobKey key(OptJob j){return {j.engine,j.workers,j.group,j.shift};}
static double quantile(std::vector<double> v,double q) {
    require(!v.empty(),"Empty measurements");std::sort(v.begin(),v.end());
    double index=q*double(v.size()-1);size_t a=size_t(index),b=std::min(a+1,v.size()-1);
    return v[a]+(index-double(a))*(v[b]-v[a]);
}
static void checkAll(id<MTLBuffer> buffer,const std::vector<uint64_t>& expected,OptJob j) {
    const uint64_t* actual=static_cast<const uint64_t*>(buffer.contents);
    for(size_t i=0;i<expected.size();++i)if(actual[i]!=(expected[i]>>j.shift))
        throw std::runtime_error(std::string(engineNames[j.engine])+" mismatch at "+std::to_string(i)+
            " threads="+std::to_string(j.workers)+" group="+std::to_string(j.group));
    for(size_t i=expected.size();i<expected.size()+16;++i)require(actual[i]==UINT64_MAX,"Guard overwritten");
}
static NSDictionary* jobJSON(OptJob j) {
    return @{@"engine":[NSString stringWithUTF8String:engineNames[j.engine]],
        @"format":j.shift?@"mantissa53":@"raw64",@"physical_threads":@(j.workers),@"threadgroup":@(j.group)};
}
int main(int argc,char**argv) {
 @autoreleasepool {
  try {
    require(argc==3,"usage: benchmark source.metal results_directory");
    const uint64_t words=8388608;const unsigned searchTrials=7,confirmationTrials=5;
    NSString* folder=[NSString stringWithUTF8String:argv[2]];
    id<MTLDevice> device=MTLCreateSystemDefaultDevice();
    require(device!=nil && device.hasUnifiedMemory,"Apple Silicon Metal required");
    NSError* error=nil;
    NSString* source=[NSString stringWithContentsOfFile:[NSString stringWithUTF8String:argv[1]] encoding:NSUTF8StringEncoding error:&error];
    require(source!=nil,"Read source: "+errorText(error));
    MTLCompileOptions* options=[[MTLCompileOptions alloc]init];options.languageVersion=MTLLanguageVersion2_4;
    std::cout<<"GPU: "<<device.name.UTF8String<<". Compilation des variantes equivalentes..."<<std::endl;
    id<MTLLibrary> library=[device newLibraryWithSource:source options:options error:&error];
    require(library!=nil,"Metal compilation: "+errorText(error));
    Pipelines pipelines(device,library);
    id<MTLCommandQueue> queue=[device newCommandQueue];require(queue!=nil,"No queue");
    conformance(device,queue,pipelines,folder);
    id<MTLBuffer> output=[device newBufferWithLength:(words+16)*8 options:MTLResourceStorageModeShared];
    require(output!=nil,"Cannot allocate output");
    std::vector<OptJob> grid;
    for(unsigned e:{10u,13u,14u,15u,16u,17u,18u,19u,1u,2u,7u})
      for(unsigned w:{8192u,32768u,65536u,131072u})for(unsigned g:{64u,128u,256u,512u})for(unsigned shift:{0u,11u})
        grid.push_back({e,w,g,shift});
    std::map<JobKey,std::vector<double>> search,confirmation;
    std::map<Family,OptJob> selected;
    std::mt19937 order(20261022);uint64_t compared=0;
    std::ofstream csv(std::string(argv[2])+"/measurements.csv");require(bool(csv),"Cannot open measurements");
    csv<<"phase,trial,seed,engine,format,logical_lanes,draws_per_lane,physical_threads,threadgroup,words,gpu_seconds,wall_seconds,ns_per_word,stored_GB_per_second,thermal_state,full_comparison_pass\n";
    NSMutableArray* thermal=[NSMutableArray array];
    auto campaign=[&](bool confirm) {
      unsigned count=confirm?confirmationTrials:searchTrials;
      for(unsigned trial=0;trial<count;++trial) {
        uint64_t seed=confirm?splitmix64(1000+trial):(trial?splitmix64(trial):42);
        std::cout<<(confirm?"Confirmation":"Recherche")<<" seed "<<trial+1<<'/'<<count<<": references CPU..."<<std::endl;
        std::map<unsigned,std::vector<uint64_t>> expected;
        for(unsigned e:{10u,1u,2u,7u})expected.emplace(e,laneReference(parameters(seed,0,words,2048,65536,0),e));
        std::vector<OptJob> jobs;
        if(confirm)for(const auto& pair:selected)jobs.push_back(pair.second);else jobs=grid;
        std::shuffle(jobs.begin(),jobs.end(),order);
        auto thermalStart=[NSProcessInfo processInfo].thermalState;
        for(auto j:jobs) {
          auto p=parameters(seed,0,words,2048,j.workers,j.shift);
          auto pipeline=pipelines.get(j.engine,p);
          if(j.group>pipeline.maxTotalThreadsPerThreadgroup) {
            require(!confirm,"Selected configuration unsupported for confirmation seed");continue;
          }
          const auto& oracle=expected.at(isMontmory(j.engine)?10:j.engine);
          uint64_t* actual=static_cast<uint64_t*>(output.contents);
          std::fill(actual+words,actual+words+16,UINT64_MAX);
          // A missing store cannot inherit a correct value from the previous job.
          for(uint64_t i=0;i<words;++i)actual[i]=~(oracle[i]>>j.shift);
          dispatch(queue,pipeline,output,p,j.engine,j.group);
          checkAll(output,oracle,j);compared+=words;
          dispatch(queue,pipeline,output,p,j.engine,j.group);
          Times t=dispatch(queue,pipeline,output,p,j.engine,j.group);
          require(t.group==j.group,"Unexpected group-size adjustment");
          checkAll(output,oracle,j);compared+=words;
          double ns=t.gpu*1e9/words;require(std::isfinite(ns)&&ns>0,"Invalid timing");
          (confirm?confirmation:search)[key(j)].push_back(ns);
          csv<<(confirm?"confirmation":"search")<<','<<trial<<','<<seed<<','<<engineNames[j.engine]<<','<<(j.shift?"mantissa53":"raw64")
             <<",2048,4096,"<<j.workers<<','<<t.group<<','<<words<<','<<std::setprecision(12)<<t.gpu<<','<<t.wall<<','<<ns<<','<<8/ns<<','<<int([NSProcessInfo processInfo].thermalState)<<",true\n";
        }
        [thermal addObject:@{@"phase":confirm?@"confirmation":@"search",@"trial":@(trial),@"start":@(thermalStart),@"end":@([NSProcessInfo processInfo].thermalState)}];
        csv.flush();require(bool(csv),"CSV write error");
        std::cout<<(confirm?"Confirmation":"Recherche")<<' '<<trial+1<<'/'<<count<<" terminee, sorties identiques a la reference."<<std::endl;
      }
    };
    campaign(false);
    std::map<Family,double> best;
    NSMutableArray* searchSummary=[NSMutableArray array];
    for(const auto& entry:search) {
      auto [e,w,g,shift]=entry.first;OptJob j={e,w,g,shift};
      // Specialized pipelines can have different resource limits for each seed.
      // Only configurations measured on every search seed can be selected.
      if(entry.second.size()!=searchTrials)continue;
      double median=quantile(entry.second,.5);Family f={e,shift};
      NSMutableDictionary* item=[jobJSON(j) mutableCopy];
      item[@"median_ns_per_word"]=@(median);item[@"q10_ns_per_word"]=@(quantile(entry.second,.1));item[@"q90_ns_per_word"]=@(quantile(entry.second,.9));
      [searchSummary addObject:item];
      if(!best.count(f)||median<best.at(f)) {best[f]=median;selected[f]=j;}
    }
    require(selected.size()==22,"Missing complete engine/format in search");
    NSMutableArray* frozen=[NSMutableArray array];for(const auto& pair:selected)[frozen addObject:jobJSON(pair.second)];
    writeJSON(folder,@"selection-before-confirmation.json",@{@"selected":frozen,@"search":searchSummary,@"rule":@"minimum median on seven search seeds, then frozen for five new seeds"});
    campaign(true);
    NSMutableArray* summary=[NSMutableArray array];
    std::cout<<"\nConfirmation sur cinq nouvelles seeds — geometrie T=2048,D=4096 inchangee\n"
             <<"moteur                         format     threads groupe   ns/mot     GB/s   ratio/v3\n";
    for(const auto& pair:selected) {
      OptJob j=pair.second;const auto& samples=confirmation.at(key(j));
      require(samples.size()==confirmationTrials,"Incomplete confirmation");
      const auto& baseline=confirmation.at(key(selected.at(Family(10,j.shift))));
      double ns=quantile(samples,.5);std::vector<double> ratios;unsigned faster=0;
      for(unsigned i=0;i<confirmationTrials;++i) {ratios.push_back(baseline[i]/samples[i]);faster+=samples[i]<baseline[i];}
      double ratio=quantile(ratios,.5);
      std::cout<<std::left<<std::setw(31)<<engineNames[j.engine]<<std::setw(11)<<(j.shift?"mantissa53":"raw64")<<std::right<<std::setw(8)<<j.workers<<std::setw(7)<<j.group
               <<std::setw(9)<<std::fixed<<std::setprecision(3)<<ns<<std::setw(9)<<8/ns<<std::setw(11)<<ratio<<'\n';
      NSMutableDictionary* item=[jobJSON(j) mutableCopy];item[@"median_ns_per_word"]=@(ns);item[@"stored_GB_per_second"]=@(8/ns);
      item[@"q10_ns_per_word"]=@(quantile(samples,.1));item[@"q90_ns_per_word"]=@(quantile(samples,.9));
      item[@"median_paired_speedup_vs_v3"]=@(ratio);item[@"seeds_faster_than_v3"]=@(faster);item[@"trials"]=@(confirmationTrials);
      [summary addObject:item];
    }
    writeJSON(folder,@"summary.json",@{@"status":@"pass",@"device":device.name,@"results":summary,
      @"interpretation":@"ratio/v3 > 1 means faster than tuned montmory_mulhi_pair on the same confirmation seeds. Five seeds give descriptive evidence, not an optimality guarantee. Write baseline is not a PRNG."});
    writeJSON(folder,@"metadata.json",@{@"device":device.name,@"os":[NSProcessInfo processInfo].operatingSystemVersionString,
      @"fully_compared_words":@(compared),@"thermal":thermal,@"logical_lanes":@2048,@"draws_per_lane":@4096,@"words":@(words),
      @"search_trials":@(searchTrials),@"confirmation_trials":@(confirmationTrials),@"timing":@"GPU generation+writes; CPU comparison and reference excluded; host dispatch+wait also recorded"});
    writeJSON(folder,@"pipeline-preparation.json",pipelines.preparation);
    std::cout<<"\nPASS. Aucun changement du flux Montmory; toutes les sorties verifiees.\n";
    return 0;
  }catch(const std::exception& e){std::cerr<<"ECHEC: "<<e.what()<<'\n';return 1;}
 }
}
