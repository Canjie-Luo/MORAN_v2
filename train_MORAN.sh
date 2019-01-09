GPU=0
CUDA_VISIBLE_DEVICES=${GPU} \
python main.py \
	--train_nips path_to_dataset \
	--train_cvpr path_to_dataset \
	--valroot path_to_dataset \
	--workers 2 \
	--batchSize 64 \
	--niter 10 \
	--lr 1 \
	--cuda \
	--experiment output/ \
	--displayInterval 100 \
	--valInterval 1000 \
	--saveInterval 40000 \
	--adadelta \
	--BidirDecoder