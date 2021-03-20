mkdir -p model/trained_with_transformations
mkdir -p model/trained_without_transformations

echo "Downloading model trained with transformations..."
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1abUF2TRucGqWhaupAO6YeK_maGqKe0Ve' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1abUF2TRucGqWhaupAO6YeK_maGqKe0Ve" -O model.tar.gz && rm -rf /tmp/cookies.txt
tar xvzf model.tar.gz -C model/trained_with_transformations/
rm model.tar.gz

echo "Model downloaded, now downloading model trained without transformations..."
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1dJJ-AQ0YRErbQ7nqaWPOQJjxINw7GmBV' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1dJJ-AQ0YRErbQ7nqaWPOQJjxINw7GmBV" -O model.tar.gz && rm -rf /tmp/cookies.txt
tar xvzf model.tar.gz -C model/trained_without_transformations/
rm model.tar.gz