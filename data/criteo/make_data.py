#!/usr/bin/python
import operator
import math

'''
raw feature size: 33763766
category number for each field
C19:2173
C18:5652
C13:3194
C12:8351593
C11:5683
C10:93145
C17:10
C16:5461306
C15:14992
C14:27
I9:104
I8:79
I1:63
I3:126
I2:113
I5:224
I4:51
I7:100
I6:148
C9:3
C8:633
C3:10131227
C2:583
C1:1460
C7:12517
C6:24
C5:305
C4:2202608
C22:18
C23:15
C20:4
C21:7046547
C26:142572
C24:286181
C25:105
I11:32
I10:9
I13:82
I12:57
'''

HEADER = 'Label,I1,I2,I3,I4,I5,I6,I7,I8,I9,I10,I11,I12,I13,C1,C2,C3,C4,C5,C6,C7,C8,C9,C10,C11,C12,C13,C14,C15,C16,C17,C18,C19,C20,C21,C22,C23,C24,C25,C26'
HEADER = HEADER.split(',')

# these field have categories over 5M
skip_header = ['C12','C16','C3','C21']
maxindex = 0
featindex = {}
featindex['none'] = maxindex
maxindex += 1
fi = open('train.txt', 'r')
print 'making feat'
for line in fi:
    row = line.replace('\n', '').split('\t')
    for j in range(1, 14):
        field = HEADER[j]
        if field in skip_header:
            continue
        value = row[j]
        if value != '':
            value = int(value)
            if value > 2:
                value = int(math.log(float(value)) ** 2)
            else:
                value = 'SP' + str(value)
        feat = field + ':' + str(value)
        if feat not in featindex:
            featindex[feat] = maxindex
            maxindex += 1
    for j in range(1 + 13, 27 + 13):
        field = HEADER[j]
        if field in skip_header:
            continue
        value = row[j]
        feat = field + ':' + value
        if feat not in featindex:
            featindex[feat] = maxindex
            maxindex += 1

print 'feature size: ' + str(maxindex)
featvalue = sorted(featindex.iteritems(), key=operator.itemgetter(1))
field_indices = {h: set() for h in HEADER[1:40] if h not in skip_header}
fo = open('featindex.txt', 'w')
for fv in featvalue:
    fo.write(fv[0] + '\t' + str(fv[1]) + '\n')
    if fv[0] != 'none':
        field = fv[0].split(':')[0]
        field_indices[field].add(str(fv[1]))
fo.close()

print 'category number for each field: '
print 'skipped fields: ' + ','.join(skip_header)
field_indices_f = open('field_indices.txt', 'w')
for key in field_indices.keys():
    print key + ':' + str(len(field_indices[key]))
    field_indices_f.write(key + '\t' + ','.join(field_indices[key]) + '\n')
field_indices_f.close()

# there is no label in test.txt, we split the train.txt for training and testing (9:1)
total = 45840617
test_begin_index =  total - 6040618
count = 0
print 'indexing for training'
fi = open('train.txt', 'r')
fo = open('train.index.txt', 'w')
fo_test = open('test.index.txt', 'w')

for line in fi:
    feats = []
    row = line.replace('\n', '').split('\t')
    for j in range(1, 14):
        field = HEADER[j]
        if field in skip_header:
            continue
        value = row[j]
        if value != '':
            value = int(value)
            if value > 2:
                value = int(math.log(float(value)) ** 2)
            else:
                value = 'SP' + str(value)
        feat = field + ':' + str(value)
        if feat not in featindex:
            feats.append('0')
        else:
            feats.append(str(featindex[feat]))
    for j in range(1 + 13, 27 + 13):
        field = HEADER[j]
        if field in skip_header:
            continue
        value = row[j]
        feat = field + ':' + value
        if feat not in featindex:
            feats.append('0')
        else:
            feats.append(str(featindex[feat]))
    if count < test_begin_index:
        fo.write(str(row[0]) + ' ' + ' '.join(feats) + '\n')
    else:
        fo_test.write(str(row[0]) + ' ' + ' '.join(feats) + '\n')
    count += 1
fo.close()
fo_test.close()
print 'Done'
