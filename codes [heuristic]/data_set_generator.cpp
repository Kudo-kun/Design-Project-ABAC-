#include <bits/stdc++.h>
#define ll long long
using namespace std;

int main() 
{
    freopen("inp_v5.txt", "w", stdout);
    ll r1 = 500;    //rand()%5+3;
    ll a1 = 10;     //rand()%5+3;
    ll c1 = 0;
    ll r2 = 75;     //rand()%5+3;
    ll a2 = 8;      //rand()%5+3;
    ll c2 = 0;
    int arr[10];    // to store no. of values for each attribute
    cout << r1 << ' ' << r2 << '\n';
    cout << a1 << ' ' << a2 << '\n';
    
    for (ll i = 0; i < a1; i++) 
    {
        ll x = rand() % 5 + 1;
        c1 += x;
        arr[i] = x;
        cout << x << " ";
    }
    
    cout << '\n' << '\n';
    for (int i = 0; i < r1; i++) 
    {
        int c2 = 0;
        // M.E. Attribute Values
        for (int j = 0; j < 5; j++) 
        {   
            int one = 0; //1 flag
            for (int k = 0; k < arr[j]; k++) 
            {
                c2++;
                int temp = (rand() % 2);
                if (one == 0 && temp == 1) 
                {
                    cout << 1 << " ";
                    one = 1;
                } 
                else if (one == 1)
                    cout << 0 << " ";
                else if (one == 0 && temp == 0) 
                    cout << 0 << " ";
            }
        }
        for (int j = c2; j < c1; j++) 
            cout << (rand() % 2) << " ";
        cout << '\n';
    }
    
    cout << '\n';
    int arr2[8];
    for (ll i = 0; i < a2; i++) 
    {
        ll x = rand() % 5 + 1;
        c2 += x;
        arr2[i] = x;
        cout << x << " ";
    }
    
    cout << '\n' << '\n';
    for (int i = 0; i < r2; i++) 
    {
        int c3 = 0;
        for (int j = 0; j < 4; j++) 
        {   
            // M.E. Attribute Values
            int one = 0; //1 flag
            for (int k = 0; k < arr2[j]; k++) 
            {
                c3++;
                int temp = (rand() % 2);
                if (one == 0 && temp == 1) 
                {
                    cout << 1 << " ";
                    one = 1;
                } 
                else if (one == 1)
                    cout << 0 << " ";
                else if (one == 0 && temp == 0) 
                    cout << 0 << " ";
            }
        }
        //Non M.E. Attribute Values
        for (int j = c3; j < c2; j++) 
            cout << (rand() % 2) << " ";
        cout << '\n';
    }
    
    cout << '\n' << '\n';
    for (int i = 0; i < r1; i++)
        for (int j = 0; j < r2; j++) 
            cout << (rand() % 8) << " \n"[j == r2-1];
    
    fclose(stdout);
    return 0;
}