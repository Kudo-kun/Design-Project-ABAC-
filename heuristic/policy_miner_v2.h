#include <bits/stdc++.h>
using namespace std;

struct ABAC_POLICY_MINER
{
    int us, obj;
    int us_at, obj_at;
    int rule_size;
    int uc, oc;

    vector<int> Uclust;
    vector<int> Oclust;
    vector<vector<int>> ACM;
    vector<vector<bool>> UAV;
    vector<vector<bool>> OAV;
    vector<pair<vector<bool>, int>> eff_rules;

    ABAC_POLICY_MINER(string path)
    {
        freopen(path.c_str(), "r", stdin);
        us_at = obj_at = 0;
        cin >> us >> obj;
        cin >> uc >> oc;
        Uclust.resize(uc);
        Oclust.resize(oc);

        for(int i = 0; i < uc; i++)
            cin >> Uclust[i], us_at += Uclust[i];
        for(int i = 0; i < oc; i++)
            cin >> Oclust[i], obj_at += Oclust[i];

    	UAV = vector<vector<bool>>(us, vector<bool>(us_at));
    	OAV = vector<vector<bool>>(obj, vector<bool>(obj_at));
        ACM = vector<vector<int>>(us, vector<int>(obj));
        
        for(int i = 0; i < us; i++)
            for(int j = 0, t; j < us_at; j++)
                cin >> t, UAV[i][j] = t;
        for(int i = 0; i < obj; i++)
            for(int j = 0, t; j < obj_at; j++)
                cin >> t, OAV[i][j] = t;
        for(int i = 0; i < us; i++)
            for(int j = 0; j < obj; j++)
                cin >> ACM[i][j];
    }

    vector<bool> intToBin(int k, int N)
    {
        vector<bool> v(N, 0);
        for(int i = 0; i < N && (k > 0); i++, k >>= 1)
            v[N-i-1] = (k & 1);
        return v;
    }

    int get_permssion(vector<bool> r)
    {
        int N = (int)r.size(), ans = 0;
        for(int i = N-3; i < N; i++)
            ans = ((ans << 1) + r[i]);
        return ans;
    }

    int rule_weight_diff(vector<bool> r1, vector<bool> r2)
    {
        int ans = 0, N = (int)r1.size();
        //check only UA and OA;
        for(int i = 0; i < N-3; i++)
            ans += (r1[i] != r2[i]);
        return ans;
    }

    bool find(vector<bool> r)
    {
        for(auto [vec, count] : eff_rules)
            if(r == vec)
                return true;
        return false;
    }

    vector<bool> append(vector<bool> v1, vector<bool> v2, vector<bool> v3)
    {
        vector<bool> res;
        for(bool it : v1)
            res.push_back(it);
        for(bool it : v2)
            res.push_back(it);
        for(bool it : v3)
            res.push_back(it);
        return res;
    }

    void merger(vector<bool> &r1, vector<bool> r2)
    {
        int N = (int)r1.size();
        for(int i = 0; i < N-3; i++)
            r1[i] = (r1[i] | r2[i]);
        return;
    }

    int attr_cluster_diff(vector<bool> r1, vector<bool> r2, int st, int len)
    {
        int diff = 0;
        for(int i = 0; i < len; i++)
            diff += (r1[st + i] ^ r2[st + i]);
        return diff;
    }

    bool can_be_merged(vector<bool> r1, vector<bool> r2)
    {
        int N = (int)r1.size(), st = 0;
        int p1 = get_permssion(r1);
        int p2 = get_permssion(r2);
        int x = 0, y = 0, z = abs(p1-p2);
        if(z > 0)
        	return false;
        //have same permissions;
        //check if they differ in exactly one user-attr-cluster;
        //x holds the number of user-clusters that have a difference;
        for(int l : Uclust)
            x += (attr_cluster_diff(r1, r2, st, l) > 0), st += l;
        //st is at the beginning index of object-attr-clusters;
        //check if they differ in exactly one object-attr-cluster;
        //y holds the number of object-clusters that have a difference;
        for(int l : Oclust)
            y += (attr_cluster_diff(r1, r2, st, l) > 0), st += l;
        //check differences;
        //if exactly one user-cluster or object-cluster differs;
        if(((x == 1) && !y) || ((y == 1) && !x))
        	return true;
        return false;
    }

    void rule_generator()
    {
        for(int i = 0; i < us; i++)
            for(int j = 0; j < obj; j++)
            {
                vector<bool> r = append(UAV[i], OAV[j], intToBin(ACM[i][j], 3));
                //present to indicate if the rule we are looking for is already present;
                bool present = find(r);
                for(int k = 0; k < (int)eff_rules.size() && (!present); k++)
                {
                    //if both rules have a one bit variation in attributes part;
                    //but have same permission, erase the rule with higher weight;
                    vector<bool> curr = eff_rules[k].first;
                    int p1 = get_permssion(r);
                    int p2 = get_permssion(curr);
                    if((p1 == p2) && (rule_weight_diff(r, curr) == 1))
                    {
                        int a = accumulate(r.begin(), r.end(), 0);
                        int b = accumulate(curr.begin(), curr.end(), 0);
                        if(a < b)
                            eff_rules.erase(eff_rules.begin() + k), k--;
                        else
                            present = true;
                    }
                }

                //second variable indicates times a rule has undergone merging;
                if(!present)
                    eff_rules.push_back({r, 0});
            }

        metrics("pre_processing");
        return;
    }

    void merge_rules()
    {
        int cap = 3;
        cout << "********* let the merging begin *********\n";    
        for(int i = 0; i < (int)eff_rules.size(); i++)
        {
            bool remove_ith = 0;
            //merge rules only if the count of merging for that rule is less than the cap
            for(int j = 0; j < (int)eff_rules.size(); j++)
                if((eff_rules[j].second <= cap) && can_be_merged(eff_rules[j].first, eff_rules[i].first))
                {
                    remove_ith = 1;
                    merger(eff_rules[j].first, eff_rules[i].first);
                    eff_rules[j].second++;
                }

            if(remove_ith == 1)
                eff_rules.erase(eff_rules.begin() + i), i--;
        }
        
        cout << "********* merging complete *********\n";
        metrics("merging");
        return;
    }

    void weight_minimizer(double percent)
    {
    	int max_rule_weight = (int)eff_rules[0].first.size();
    	int bound = (percent * max_rule_weight);
    	cout << "bound = " << bound << '\n';
    	for(int i = 0; i < (int)eff_rules.size(); i++)
    		if(accumulate(eff_rules[i].first.begin(), eff_rules[i].first.end(), 0) > bound)
    			eff_rules.erase(eff_rules.begin() + i), i--;

    	metrics("minimizing");
        return;
    }

    void metrics(string action)
    {
    	int sum = 0;
    	cout << "count after " << action <<  " = " << eff_rules.size() << '\n';
        for(auto [rule, count] : eff_rules)
            for(bool it : rule)
                sum += it;
        cout << "avg. rule weight: " << (double)sum/(double)eff_rules.size() << '\n';
        cout << "total weight after " << action << " = " << sum << '\n';
        return;
    }
};

