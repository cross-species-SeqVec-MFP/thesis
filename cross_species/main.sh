
seqvecOutputPath=$1;
fastaPath=$2;
embeddingsPath=$3;
evidenceCodesPath=$4;
goTermsPath=$5;
goatoolsPath=$6;
dataPath=$7;
ontology=$8;
mlpPath=$9;
predictionsPath=$10;
blastPath=$11;

# step 1 - run seqvec code from heizinger et al  ??? link ???, save at seqvecOutputPath
## FILL THIS IN

# step 2 - save embeddings in necessary format
python extract_embeddings.py $seqvecOutputPath $fastaPath $embeddingsPath;

mkdir $goTermsPath
# step 3
python evidence_codes.py $embeddingsPath $evidenceCodesPath $goTermsPath $fastaPath;


# step 4 install GOAtools ??? link ??? in directory goatoolsPath
while IFS= read -r line; do
    $goatoolsPath"/bin/wr_hier.py" $line --concise --up -o $goTermsPath'/GO_ancestors/'$line'.txt'
done < $goTermsPath'/list_terms.txt';

mkdir $dataPath;
#step 5 - get class labels
python GO_statistics.py $evidenceCodesPath $fastaPath $dataPath $goTermsPath;

# step 6a - split training data
python split_stratify.py $dataPath;

# step 6b - split training data
python Writing_fasta_multilabel.py $dataPath $fastaPath;

# step 7 - get common go terms
mkdir -p $goTermsPath'index_term_centric/';
mkdir -p $goTermsPath'index_protein_centric/';
python get_GO_terms.py $dataPath $goTermsPath;

# step 8 get path length to root for each term
python GO_level.py $ontology $goTermsPath;

# step 9 train MLP
mkdir -p $mlpPath'/epochs';
python neural_network.py $ontology $dataPath $goTermsPath $mlpPath;

mkdir -p $predictionsPath;
# step 10 evaluate MLP
python neural_network_results.py $ontology $dataPath $goTermsPath $mlpPath $predictionsPath;


# step 11 - run psi blast
makeblastdb  -in $fastaPath'/mouse_train_proteinsequence.fasta' -dbtype prot -title "BLAST DB mousemodel" -parse_seqids -out $blastPath'/BLAST_mousemodel';
mkdir -p $blastPath'/predictions/';
for species in 'mouse_valid' 'mouse_test' 'rat' 'human' 'zebrafish' 'celegeans' 'yeast' 'athaliana';
do

psiblast -query $fastaPath'/'$species'_proteinsequence.fasta' -outfmt 7 -db $blastPath'/BLAST_mousemodel' -out $blastPath'/predictions/psiBLAST_'$species -num_iterations 3
done;

# step 12 - evaluate psi blast
python get_annotations_psiblast.py $ontology $dataPath $goTermsPath $blastPath $predictionsPath;

# step 13 - get conserved functions
python GO_term_preserved.py $goTermsPath $predictionsPath;
