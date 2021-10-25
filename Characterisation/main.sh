# step 0 create your own dataset - see README


trainPath=$1;
modelPath=$2;
validPath=$3;
testPath=$4;
goatoolsPath=$5;
lengthPath=$6;
testProteinNames=$7;

mkdir -p $modelPath

# step 1 train logistic regression with different parameters
python LR_train.py $trainPath $modelPath;

# step 2 evaluate
python LR_results.py $validPath $testPath $modelPath;

# step 3 get term path length to root
python GO_level.py $trainPath $validPath $testPath;

# step 4 install GOAtools ??? link ??? in directory goatoolsPath
while IFS= read -r line; do
    $goatoolsPath"/bin/wr_hier.py" $line --concise -o './GO_descendants_'$line'.txt'
done < './all_terms.txt';


# step 5 term-centricperformance characterization
python term_centric.py $modelPath;


# step 6 protein-centricperformance characterization
python protein_centric.py $modelPath $lengthPath $testPath;

# step 7 domain/family/superfamily analysis
python protein_domains.py $testPath $testProteinNames $modelPath
