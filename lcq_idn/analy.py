import random
path = 'error_lines.txt'
with open(path, 'r') as f:
        lines = f.readlines()

labels = [0,0]
no = [0]*51
error_data = {}
for line in lines:
    refer, test, label = line.split()
    labels[int(label)] += 1
    re = refer.split('/')[0]
    no[int(re)-50] += 1

    if re not in error_data:
          error_data[re] = []
    error_data[re].append(line.split('\n')[0])

for i in range(len(no)):    
    print(i+50,no[i],'-------',end="")
print(labels) #[3887, 529]
#print(error_data['051'],len(error_data['051']))


# path = '/home/linchaoqun/project/fuji37450_IDN/dataset/BHSig260/Bengali_resize/test_all_pairs.txt'
# data = {}
# with open(path, 'r') as f:
#         lines = f.readlines()
# for line in lines:
#     refer, test, label = line.split()

#     if re not in error_data:
#           error_data[re] = []
#     error_data[re].append(line)

all_data = []
for i in range(51,101,1):
    single_data = []
    single_data_true = []
    single_data_false = []
    true_data = []
    fake_data = []
    for j in range(1,25,1):
        true_data.append('{:03d}/B-S-{:03d}-G-{:02d}.tif'.format(i,i,j))
    for j in range(1,31,1):
        fake_data.append('{:03d}/B-S-{:03d}-F-{:02d}.tif'.format(i,i,j))
  #
    for td in range(len(true_data)):
        for ttd in range(td+1,len(true_data)):
            single_data_true.append(true_data[td]+" "+true_data[ttd]+" 1")
        for fd in range(len(fake_data)):
            single_data_false.append(true_data[td]+" "+fake_data[fd]+" 0")
    
    for fal in single_data_false:
        if '{:03d}'.format(i) not in error_data:
            continue
        if(fal in error_data['{:03d}'.format(i)]):
            single_data_false.remove(fal)

    if (len(single_data_false)<276):
        single_data_false += random.sample(error_data['{:03d}'.format(i)], 276-len(single_data_false))

    print(len(single_data_false),len(single_data_true))

    single_data_false = random.sample(single_data_false, 276)
    all_data = all_data + single_data_true + single_data_false
    
train_data_path = r'test_lcq_pairs.txt'
file2 = open(train_data_path,'w+')
for aaa in all_data:
    file2.write(aaa+"\n")
file2.close()




# path = '/home/linchaoqun/project/fuji37450_IDN/dataset/BHSig260/Bengali_resize/test_lcq_pairs.txt'
# data = {}
# with open(path, 'r') as f:
#     for range
#         f.write()