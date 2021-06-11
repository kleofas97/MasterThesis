import editdistance

recognizedset = open('output.csv')
gtset = open('tests/data_for_pipeline_testing/gt.txt')
numCharErr = 0
numCharTotal = 0
print('Ground truth -> Recognized')
sample = 0
for recongnized, gt in zip(recognizedset, gtset):
    recongnized = recongnized[1:]
    recongnized = recongnized.replace('||','|')
    lineSplit = gt.strip().split(' ')
    gt = ' '.join(lineSplit[8:])
    dist = editdistance.eval(recongnized, gt)
    numCharErr += dist
    numCharTotal += len(gt)
    print('[OK]' if dist == 0 else '[ERR:%d]' % dist, '"' + gt + '"', '->',
          '"' + recongnized + '"')
    sample +=1
charErrorRate = numCharErr / numCharTotal
print(f'Character error rate: {charErrorRate * 100.0}%.')
print("number of lines: {}".format(sample))
