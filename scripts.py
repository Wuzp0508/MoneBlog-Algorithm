import sys
from text_classification.TextClassfication import text_classification
from text_classification.LLMTextClassification import LLM_text_classification
from animal_recognition.AnimalRecognition import animal_recognition
from animal_recognition.LLMAnimalRecognition import LLM_animal_recognition

# sys.stdout.reconfigure(encoding='gbk')

if __name__ == '__main__':
    # print(sys.argv[1], sys.argv[2])
    if sys.argv[1] == '0':
        print('开始文本情感识别...')
        Category = ['消极']
        textTotal = ''
        for (i, text) in enumerate(sys.argv[2:]):
            textTotal += text
            # print(f'-------第 {i+1} 条文本：{text}-------')
        result0 = text_classification(textTotal)
        result = LLM_text_classification(textTotal, prompt='情感类别：开心,喜悦,中性')
        result = result.split('\n')  # 取最后一行，即情感分类结果
        if (result[0] in Category) and result0 in Category:
            print(result0)  
            print(result[1][3:])
        else:
            print(result[0])  
            print(result[1][3:])            
        

    elif sys.argv[1] == '1':
        print('开始图像动物检测...')
        Category = ['cat', 'dog']
        for (i, image) in enumerate(sys.argv[2:]):
            # print(f'-------第 {i+1} 个图像：{image}-------')
            print('图片: ' + image)
            result0 = animal_recognition(image)
            result1 = LLM_animal_recognition(image)
            if isinstance(result1, list):
                flag = True
                for category in Category:
                    result1Array = result1[0].split('\n')
                    if (category in result1Array[1]) or (category in result0):
                        print('预测类别: '+ category)
                        print('位置：'+str(result1Array[2])+" shape: "+str(result1[1].shape))
                        # print(f"![image]({URL})")# 替换为URL
                        flag = False
                        break
                if flag:
                    print('预测类别: other')
                
            else:
                print(result0)

    elif sys.argv[1] == '2':
        print('开始基于大模型的文本情感识别...')
        for (i, text) in enumerate(sys.argv[2:]):
            print(f'-------第 {i+1} 条文本：{text}-------')
            result = LLM_text_classification(text)
            print(result)

    elif sys.argv[1] == '3':
        print('开始基于大模型的图像动物检测...')
        for (i, image) in enumerate(sys.argv[2:]):
            print(f'-------第 {i+1} 个图像：{image}-------')
            result = LLM_animal_recognition(image)
            print(result)
