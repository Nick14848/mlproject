from setuptools import find_packages, setup
from typing import List

HYPEN_E_DOT = "-e ."
def get_requirements(file_path:str)->List[str]:
    '''
    This function will return a list of requirements
    '''
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n", "") for req in requirements] # 移除 \n 换行
        
        # 移除 requirement.txt 最后的 -e .
        # -e . 的作用是把 requirements.txt 和 setup.py 链接在一起
        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
    return requirements



setup(
    name = 'mlproject',
    version = '0.0.1',
    author = 'Nick_Tsai',
    author_email = 'nicktsai1221@gmail.com',
    packages = find_packages(),
    install_requires = get_requirements("requirements.txt")       # ['pandas', 'numpy', 'seaborn']
)