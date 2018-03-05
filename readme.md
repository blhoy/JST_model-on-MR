> JST_pre.py：
>
> 主程序为JST_pre.py,使用命令python JST_pre.py num运行各类先验信息：
>
> num 可选0~4,0表示无先验（empty_prior），1表示使用mpqa先验（mpqa由JST模型提出者提供的先验词集合），2表示paradigm_words，3表示全部主观性词典（full_subjectivity_lexicon），4表示筛选过的主观性词典（filter_lexicon），五个文件在该目录下均由get_constraint.py得到。
>
> JST_pre中__init__函数可以调整实验参数。



> preprocessing.py：
>
> 文档中neg与pos为原始文本，使用preprocessing.py处理，去除非字母的词，并利用目录下的stop_words.txt去除停用词，最终得到MR文件，文档中的MR.dat文件为JST模型提出者提供的预处理文件。



> get_constraint.py：
>
> 文档中使用get_constraint.py可以得到经过处理的先验词典，后缀名均为.constraint，subjclueslen.tff为MPQA最新的主观性词典，处理可得到full_subjectivity_lexicon.constraint与filter_lexicon.constraint，mpqa.constraint由JST作者提供。



> _JST.pyx：
>
> 使用cython实现的gibbs采样部分，windows下和linux下的编译链接到python的动态库_JST.pyd、_JST.so已提供。