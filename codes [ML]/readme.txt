obj_data.csv - objects in the system along with attributes
sub_data.csv - subjects/users in the system along with attributes
pol_data (v1, v2, v3) - 3 different policies (with 10%, 12.3%, 12.6% +ve rules)

To create clean the rules and convert them to categorical format:
1. Run the abac.py to expand all the rules. This will generate an exhaustive ACM. 
   Do make sure the correct obj_data, sub_data and pol_data files are being read to generate the ACM.
2. To generate the dataset for classification, run train_cleaner.py. 
   The flag -i indicates the input ACM file and -o will denote the output file to generated.
   Ex: python train_cleaner.py -i ACM-v3.txt -o abac-v3.txt
3. The dataset file is generated in the ML folder.

To run the classification testing: python run_classify.py -i abac-v3.txt; here -i denotes the input dataset file.
  