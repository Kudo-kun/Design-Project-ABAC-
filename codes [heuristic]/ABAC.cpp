#include "policy_miner.h"
#include <chrono>
using namespace std;
using namespace std::chrono; 


int main()
{
    //compile with -O2/-O3 flag for best results;
    ABAC_POLICY_MINER apm = ABAC_POLICY_MINER("inp_v5.txt");
    cout << "processing...\n";
    auto start = high_resolution_clock::now(); 
    
    apm.rule_generator();
    apm.merge_rules();
    
    //apm.weight_minimizer(0.70);
    // for(auto rule : apm.eff_rules)
    // {
    // 	for(bool it : rule)
    // 		cout << it;
    // 	cout << '\n';
    // }
   
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<seconds>(stop - start); 
    cout << "execution time: " << duration.count() << " seconds\n";
    return 0;
}