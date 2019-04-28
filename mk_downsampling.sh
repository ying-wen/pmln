advertisers="1458 2261 2997 3386 3476 2259 2821 3358 3427 all"


for advertiser in $advertisers; do
    echo $advertiser
    python make_downsampling_data.py ./data/make-ipinyou-data/$advertiser/train.yzx.txt
done

echo "criteo"
python make_downsampling_data.py ./data/criteo/train.index.txt