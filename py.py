from dog_classify.engine import labels
for i in range(len(labels['id'])):
    if(labels['breed'][i]=='dingo'):
        print('train/'+labels['id'][i]+'.jpg')