from utils.save import save_model

saves = [[0,0,0]] * 5
score = [1,2,4,3,5,9,7,8,7,6,9,6,8,8]
num = len(score)
step = list(range(0, 2000*14, 2000))
for i in range(len(step)):
    saves = save_model(score[i], step[i], 0, saves, log_path='./test')
    print(saves)