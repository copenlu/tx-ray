mkdir -p data/
cd data/
wget -c https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
tar -xvzf aclImdb_v1.tar.gz
wget -c https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-raw-v1.zip
unzip wikitext-2-raw-v1.zip
