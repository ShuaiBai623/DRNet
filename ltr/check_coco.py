from ltr.dataset import  MSCOCOSeq
coco_train = MSCOCOSeq()
train_frame_ids =[1,2,3]
for i in range(148924,coco_train.get_num_sequences()):
	print(i)
	coco_train.get_frames(i, train_frame_ids)