from flask import Flask, request, render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

# 创建了一个 Flask 应用实例
# 传递 __name__ 给 Flask 是为了让它知道在哪里查找静态文件和模板文件
app = Flask(__name__)

# Route for a home page
# 告诉 Flask 当用户访问特定的 URL 时，应该触发哪个函数
@app.route('/') # 根 URL（即 /，通常是主页）
def index():
    # render_tempalte 渲染 HTML 模板
    # Flask 默认会在 templates 目录下查找 HTML 文件
    return render_template('index.html')

# 定义新的路由 /predictdata，支持 GET 和 POST 方法
# GET 请求：如果用户访问这个 URL 而没有提交表单，程序返回 home.html，显示输入表单
# POST 请求：如果用户提交表单（即 POST 请求），程序将获取表单中的数据，进行预测，然后返回结果
@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET': # GET method
        return render_template('home.html')
    else:
        # 将表单数据封装为一个自定义的 CustomData 对象
        data=CustomData(
            # 从 HTML 表单中获取用户输入的值
            gender=request.form.get('gender'),
            race_ethnicity=request.form.get('ethnicity'),
            parental_level_of_education=request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course'),
            reading_score=float(request.form.get('writing_score')),
            writing_score=float(request.form.get('reading_score'))
        )
        
        # 将输入数据转为 Pandas DataFrame
        pred_df=data.get_data_as_data_frame()
        print(pred_df)
        print("Before Prediction")
        
        # 加载机器学习的预测管道（模型和处理流程），并对输入数据进行预测
        predict_pipeline=PredictPipeline()
        print("Mid Prediction")
        results=predict_pipeline.predict(pred_df)
        print("after Prediction")
        # 将预测结果渲染到 home.html 模板中，显示给用户
        return render_template('home.html',results=results[0]) # results 是 list format
    
# 启动 Flask 开发服务器
if __name__=="__main__": # 如果当前文件是作为主程序运行，就启动 Flask 开发服务器
    # 将服务器绑定到所有 IP 地址
    app.run(host="0.0.0.0")    

'''
__name__ 是 Python 的一个特殊变量，用来识别当前模块的名称
如果一个 Python 文件是作为脚本直接运行的, __name__ 的值就是 "__main__"
如果这个 Python 文件是作为模块被导入到其他文件中, __name__ 就会是这个模块的名字（通常是文件名，去掉 .py)

if __name__ == "__main__" 的作用是确保某些代码（如启动 Flask 服务器的代码）只会在文件被直接运行时执行，
而不会在文件被导入到其他模块时执行，避免不必要的行为或冲突
'''
