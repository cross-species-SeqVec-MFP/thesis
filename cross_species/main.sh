
seqvecOutputPath=$1;
fastaPath=$2;
embeddingsPath=$3;
evidenceCodesPath=$4;
goTermsPath=$5
goatoolsPath=$6
dataPath=$7

# step 1 - run seqvec code from heizinger et al  ??? link ???, save at seqvecOutputPath
## FILL THIS IN

# step 2 - save embeddings in necessary format
python extract_embeddings.py $seqvecOutputPath $fastaPath $embeddingsPath;

# step 3
python evidence_codes.py $embeddingsPath $evidenceCodesPath $goTermsPath $fastaPath;


# step 4 install GOAtools ??? link ??? in directory goatoolsPath
while IFS= read -r line; do
    $goatoolsPath"/bin/wr_hier.py" $line --concise --up -o $goTermsPath'/GO_ancestors/'$line'.txt'
done < $goTermsPath'/list_terms.txt';


#step 5 - get class labels
python GO_statistics.py $evidenceCodesPath $fastaPath $dataPath $goTermsPath;

# step 6 - split training data
python split_stratify.py $dataPath;

python Writing_fasta_multilabel.py $dataPath $fastaPath;

# step 7 - get common go terms
mkdir -p $goTermsPath'index_term_centric/';
mkdir -p $goTermsPath'index_protein_centric/';
python get_GO_terms.py $dataPath $goTermsPath;
