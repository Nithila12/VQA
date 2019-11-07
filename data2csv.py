import csv
import json

file = './Questions_Train_mscoco/OpenEnded_mscoco_train2014_questions.json'
fil2 = './Annotations_Train_mscoco/mscoco_train2014_annotations.json'

with open(file,'r') as f:
    dataQ = json.load(f)
    
with open(fil2,'r') as f:
    dataA = json.load(f)

CSV = [["imagepath",'question','answer']]
for i,q in enumerate(dataQ['questions']):
    qid = q['question'].replace("?"," ?")
    ans = dataA['annotations'][i]['answers'][0]['answer']
    imgid = q['image_id']
    imgpath = f"./train2014/COCO_train2014_{str(imgid).zfill(12)}.jpg"
    CSV.append([imgpath, qid, ans])


with open('LudwigVQA.csv','w') as ldcsv:
    writer = csv.writer(ldcsv, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    for k in CSV:
        writer.writerow(k)