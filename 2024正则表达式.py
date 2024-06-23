# str = input()
# arr = str.split(' ')
# print(arr)
#
# n = len(arr) - 1
# print(n)
#
# print(len(arr[n]))

# str =input()
# list = str.split(' ')
# list.reverse()
# print(list)
# print(' '.join(list))
# abcd 12345 ed 125 ss 123456789

# str = input()
# words= str.split('')
#
# for word in words:
#     a = max(len(word))
#     b = []
#     b.append(a)
#     if len(word) > max(b):
#         print(word)
#     if len(word)<max(b):
#         continue


# words = ['abc', '12345', '123', 'abcdefj']
#
# # 初始化2个变量：一个用于存储最长字符串的长度，另外一个存储最长字符串
# max_length = 0
# longest_word = ''  # 定义初始的字符串的方法
#
# # 如果当前字符串的长度大于已知的最长长度
# for word in words:
#     if len(word) > max_length:
#         # 更新最长长度和最长字符串
#         max_length = len(word)
#         longest_word = word
#
# print(longest_word)

# str_input = input()
# letters = []
# digits = []
#
# for char in str_input:
#     if char.isalpha():
#         letters.append(' '.join(char))
#     elif char.isdigit():
#         digits.append(' '.join(char))
#
# print("Letters:", letters)
# print("Digits:", digits) and str_input[i].isalnum()

# #  bc.d.12345.ed125e --> bc.d. 12345 .ed 125 e (数字干净在一起，非数字在一起）
# str_input = input()
# result = ''
# for i in range(len(str_input)):
#  if i > 0 and str_input[i - 1].isalpha() != str_input[i].isalpha() and str_input[i].isalnum() :  # 当前一个和后一个
#     result += ' '
#     if i > 0 and (str_input[i - 1].isalnum() and str_input[i].isalnum()) == False:
#         result += ' '  # bc.d.12345.ed125ss123456789
#     result += str_input[i]
#
# # 去除字符串开头的空格（如果有的话）
# if result.startswith(' '):
#     result = result[1:]
# print(result)

# i = 0
# str_input = input()
# for i in range(len(str_input)):
#     if str_input[i].isalnum():
#      print(str_input[i].isalnum())

# def split_number_and_non_number(s):
#     result = [] # 注意result和current_word的不同
#     current_word = ''  # 用于累积当前的数字串，有清空操作
#     preceding_is_number = False
#     longest_word = ''
#
#     for char in s:
#         if char.isdigit():
#             if not preceding_is_number:
#                 # 如果前一个字符不是数字，并且结果不是空的，则添加空格
#                 if result and not result[-1].isdigit():
#                     result.append(' ')
#                 preceding_is_number = True # 设置标志变量为True
#             current_word += char  # 累积当前数字
#
#         else:
#             if preceding_is_number:
#                 # 如果前一个字符是数字，则 1.添加当前数字字符串和空格到结果中 2.对数字串进行长短比较
#                 result.append(current_word)
#                 result.append(' ') #1
#                 #2 更新最长数字串
#                 if len(current_word) > len(longest_word):  # 因为是数字，也可以直接比大小
#                     longest_word = current_word
#
#                 preceding_is_number = False
#                 # 重置当前数字字符串
#                 current_word = ''
#             result.append(char)
#
#             # 如果字符串以数字结尾，确保将其添加到结果中
#     if preceding_is_number:
#         result.append(current_word)
#
#         if len(current_word) > len(longest_word):  #  循环后才有最后一个数字串，但比较是在循环里面做的，这里补一次
#             longest_word = current_word
#
#     return ''.join(result), longest_word
#
# # 示例输入
# input_string = input() # 负号、浮点不行
# output_string, longest_word = split_number_and_non_number(input_string)
# print(output_string)
# print(longest_word)

# def split_number_and_non_number(s):
# result = []
# current_word = ''
# longest_word = ''
#
# for char in s:
#     if char.isdigit() or char == '-' and not current_word:  # 允许负号开头
#         current_word += char
#         # 这里不进行比较，因为current_word可能还在累积中
#
#     else:
#         # 如果前一个字符是数字，并且current_word不为空
#         if current_word:
#             result.append(current_word)
#             # 在这里添加空格，因为我们已经完成了current_word的累积
#             result.append(' ')
#
#             # 更新最长数字串
#             if current_word > longest_word:
#                 longest_word = current_word
#
#             current_word = ''  # 重置current_word
#
#         # 添加非数字字符到结果中
#         result.append(char)
#
#         # 如果字符串以数字结尾，确保将其添加到结果中，并比较长度
# if current_word:
#     result.append(current_word)
#     if len(current_word) > len(longest_word):
#         longest_word = current_word
#
# return ''.join(result), longest_word


# # 示例输入
# input_string = input()  # 现在应该能够处理负号了
# # 调用函数并打印结果
# output_string, longest_word = split_number_and_non_number(input_string)
# print(output_string)
# print(longest_word)


# def long_num(str):
#     max_list = []
#     for i in range(len(str)):
#         new_list = []
#         for j in range(i+1,len(str)):
#             if str[j].isdigit():
#                 new_list.append(str[j])
#             else:
#                 break
#         if len(max_list) < len(new_list):
#             max_list = new_list
#     return ''.join(max_list)
# str = input()
# print(long_num(str))
#
# import re
# w = re.findall(r'\d+', input())
# n = list(map(int, w))
# print(max(n))

import re

# w = re.findall('\d+', input())
# n = list(map(int, w))
# print(max(n))


# w = re.findall('\d{4}', input())
# n = list(map(int, w))
# print(max(n))

# import re
# print(re.match('www', 'www.runoob.com').span())  # 在起始位置匹配
# print(re.match('com', 'www.runoob.com'))         # 不在起始位置匹配

# phone = "2004-959-559  # 这是一个国外电话号码"

# 删除字符串中的 Python注释
# num = re.sub(r'#.*$', "", phone)
# print("电话号码是: ", num)
#
# # 删除非数字(-)的字符串
# num = re.sub(r'\D', "", phone)
# print("电话号码是 : ", num)


# def double(matched):
#     value = int(matched.group('value'))
#     return str(value * 2)
#
# s = 'A23G4HFD567'
# print(re.sub('(?P<value>\d+)', double, s))

# a = 'EXCEL12233,dbad111c,PPT233ababcd'
# d = re.findall('[a-zA-Z]{3,5}', a)
# print(d)
#
# a = 'EXCEL12233,dbad111c,PPT233ababcd'
# e = re.findall('[a-zA-Z]{3}', a)  # '同非贪婪[a-zA-Z]?'
# print(e)
#
# a = 'EXCEL12233,EXCE111c,PPT233aexceeee'
# f = re.findall('EXCe*', a, re.I)  # '同非贪婪[a-zA-Z]?'
# print(f) print('{:-^20}'.format('result'))

# 取出abcabc
# c = 'abcabcewwqabcabcabc222iioooabc'
# g = re.findall('abcabc', c) # 列表元素
# e = re.findall('(abc)(abc){2}', c) # ()()是当一个元组被取出
#
# print(g)
# print('{:-^20}'.format('e'))
# print(e)


# def hi(c):
#     return 'hello'
#
# a = 'hihi123EE123'
# b = re.sub('EE', hi, a, 1)
# print(b)

# # 提取电子邮件地址的用户名和域名部分。
# text = "Contact us at info@example.com or support@anotherdomain.net for assistance."
# 
# pattern = r'(\w+)@(\w+\.\w+)' # 邮箱地址 info@example.com   support@anotherdomain.net
# matches = re.findall(pattern,text)  # @不被捕获
# # [('info', 'example.com'), ('support', 'anotherdomain.net')]
# 
# for match in matches:
#     username,domain =match
#     print(f'Username:{username},Domain:{domain}')
# 
# # 或者，如果你只想使用 re.search 或 re.match（它们返回单个匹配项作为 Match 对象）
# match = re.search(pattern, text)
# if match:
#     username = match.group(1)  # 第一个捕获组（索引从1开始）
#     domain = match.group(2)    # 第二个捕获组
#     print(f"Username: {username}, Domain: {domain}")

# a='abcFBIabcFBIaFBIabc'
# def 函数名(b):
#     分段获取 = b.group()
#     return '$'+分段获取+'$'
# r=re.sub('FBI',函数名,a)
# print(r)

# import re
#
# a = 'ab6543Iab78890a87abc'
#
#
# def replace(match):
#     分段获取 = match.group()  # 分段获取str
#
#     if int(分段获取) >= 5:  # str变成int
#         return '9'
#     else:
#         return '0'
#
#
# r = re.sub('\d', replace, a)  # 被替换的数字部分\d
# print(r)


# a ='0344-67796666'
# b = re.compile(r'\d{4}-\d{8}')
# r =re.findall(b,a)
# print(r)
#
# a ='0344-67796666'
# r =re.findall(r'\d{4}-\d{8}',a)
# print(r)

# a='life is short ,i use python'
# r =re.findall('life(.*)python',a)
# print(r)

# content = '''
# 薪资：2万/月，职位：市场部经理，公司信息：知名快消品企业，市场影响力大。
# 工资：4万/每月，职位：首席执行官，公司信息：国际集团企业，业务遍布全球。
# 工资：1.3万元/每月，职位：软件工程师，公司信息：创新科技公司，专注于软件开发领域。
# 薪资：3.5万元/每月起，职位：研发总监，公司信息：高新技术企业，专注于产品研发。
#
# 薪资：10000/每月，职位：数据分析师，公司信息：专业数据分析机构，提供精准数据服务。
# 薪资：18000/月，.职位：产品经理，公司信息：创新产品公司，致力于打造优质产品
#
# 薪资：5000-8000元/月，职位：行政助理，公司信息：综合性企业，提供完善的行政支持。
# 工资：25000-35000元/每月，职位：销售总监，公司信息：行业领先的营销公司，业绩突出。
# 工资：15000-20000元/月，职位：人力资源主管，公司信息：大型互联网企业，提供多元化职业发展。
#
# 工资：150,500,000元/月，职位：首席执行官，公司信息：大型互联网企业，提供多元化职业发展。
# 工资：面议/每月，职位：创意总监，公司信息：广告传媒公司，创意无限。
# '''
# # r = re.compile(r'([\d.]+)万/每{0,1}月')
# # for one in r.findall(content):
# #     print(one)
#
# r = re.compile(r'(\d+)?-?(\d+(\.\d+)?)万?元?/每?月')
# e = re.findall(r'(\d+)?(-?)(\d+(\.\d+)?)万?元?/每?月',content)
# # for one in r.findall(content):
# #     print(one)
# print(e)


# names = '关羽; 张飞, 赵云,马超, 黄忠  李逵'
# namelist = re.split(r'[;,\s]\s*',names)
# print(namelist)


# import re
#
# names = '''
#
# 下面是这学期要学习的课程：
#
# <a href='https://www.bilibili.com/video/av66771949/?p=1' target='_blank'>点击这里，边看视频讲解，边学习以下内容</a>
# 这节讲的是牛顿第2运动定律
#
# <a href='https://www.bilibili.com/video/av46349552/?p=125' target='_blank'>点击这里，边看视频讲解，边学习以下内容</a>
# 这节讲的是毕达哥拉斯公式
#
# <a href='https://www.bilibili.com/video/av90571967/?p=33' target='_blank'>点击这里，边看视频讲解，边学习以下内容</a>
# 这节讲的是切割磁力线
# '''
#
# # 替换函数，参数是 Match对象
# def subFunc(match):
#     # Match对象 的 group(1) 返回的是第一个group分组的内容
#     number = int(match.group(1)) + 6
#     dest = f'/av{number}/'
#     # 返回值就是最终替换的字符串
#     return dest
#
# newStr = re.sub(r'/av(\d+)/', subFunc , names)
# print(newStr)
#
#
# re.search()
#
# content = '''hello world hi hello xiao12233\njello world hi hi xiao12233'''
#
# # 多行模式会识别字符串的每一行，分别进行匹配
# r = re.findall(r'^[hj]ello.*12233$', content, re.M)
#
# # 单行模式用re.DOTALL启动，^和$只匹配整个字符串的开头和结尾，而不是每行的开头和结尾。
# s = re.findall(r'^[hj]ello.*12233$', content, re.DOTALL)
# e = re.findall(r'^[hj]ello.*12233$', content, re.S)
#
# print('{:-^30}'.format('多行-跨行n'))
# print(r)
# print('{:-^30}'.format('单行-不跨n'))
# print(s)
# print(e)

# source = '''001-苹果价格-60
# 002-橙子价格-70
# 003-香蕉价格-80
# '''
#
# p = re.compile(r'^\d+', re.M)
# q = re.compile(r'^\d+')
# for i in p.findall(source):
#     print(i)
# for j in q.findall(source):
#     x = print(j)

# num = '1123455641'
# pattern = re.compile(r'(\d{1,3}(?=(?:\d{3})+$))',num)
# #
# replace = lambda match: ',' if match.group() else ''
# r= re.sub(pattern,replace,num)
# print(r)

# import re
#
# num = '1123455641'
#
# # 编译一个正则表达式，匹配从字符串开始到第一个三位数字组，
# # 然后匹配后续的每三位数字组（可能不是每组都完整存在）
# # 使用非捕获组来分隔每三位数字，并在替换时添加逗号
# pattern = r'(?:\d(?<!\d)\d{0,2}(?=(\d{3})+$)|(?<=\d)\d{3}(?!\d))'
#
# # 定义一个替换函数，该函数将在每三位数字后添加逗号
# # 注意：这里使用了一个简单的lambda表达式而不是命名函数
# 我们不使用match.group()来获取匹配内容，因为断言不会捕获内容
# # 我们只是简单地添加逗号
# replace_with_comma = lambda match: ',' if match.group() else ''
#
# # 使用re.sub替换匹配的部分（但实际上我们是在断言的位置添加逗号）
# # 注意：这里我们传递的是模式字符串而不是编译后的对象
# r = re.sub(pattern, replace_with_comma, num,  re.DOTALL)
#
# # 移除开头的逗号（如果有的话）
# if r and r[0] == ',':
#     r = r[1:]
#
# # 打印结果
# print(r)  # 输出应该是: 1,123,455,641

# import beautifulsoup4
# context1 = '''<div class="jsx-156868388 tabs"><header class="jsx-156868388">
# <div class="jsx-1323332218 highlight"></div>
# <div class="jsx-156868388 scroll-container hide-divider">
# <div role="button" data-geist="tab-item" class="jsx-729898008 tab active">Legends</div>
# <div role="button" data-geist="tab-item" class="jsx-2238088534 tab disabled">Edit</div>
# <div role="button" data-geist="tab-item" class="jsx-853920892 tab">Test</div>'''
#
# t = re.findall(r'<div[^>]*>[^<>]*(((?'group'<div[^>]*>))+((?'-group'</div>)[^<>]*)+)*(?(Open)(?!))</div>', context1)
# print(t)

# from bs4 import BeautifulSoup
#
# html_content = '''
# <html>
# <body>
#     <div class="jsx-156868388 tabs"><header class="jsx-156868388">
#     <div class="jsx-1323332218 highlight"></div>
#     <div class="jsx-156868388 scroll-container hide-divider">
#     <div role="button" data-geist="tab-item" class="jsx-729898008 tab active">Legends</div>
#     <div role="button" data-geist="tab-item" class="jsx-2238088534 tab disabled">Edit</div>
#     <div role="button" data-geist="tab-item" class="jsx-853920892 tab">Test</div>
# </body>
# </html>
# '''
#
# # 使用 BeautifulSoup 解析 HTML 内容
# soup = BeautifulSoup(html_content, 'html.parser')
#
# # 查找所有的 div 标签
# divs = soup.find_all('div')
#
# # 遍历并打印每个 div 标签的内容
# for div in divs:
#     print(div.prettify())  # prettify() 函数可以格式化输出 HTML
# print("-------------------------------")
# # 或者，如果你只想获取 div 标签内的文本内容
# for div in divs:
#     print(div.get_text())
#
# a = 'EXCEL12233,EXCE111c,PPT233aexceeee'
# f = re.findall('EXCe*', a, re.I)  # '同非贪婪[a-zA-Z]?'
# print(f)


# content1 = 'abcadcewwqabcadcabcabc222iioooabc'
# content2 = 'abcabcewwqabcabcadcabcabcadc222iioooabc'
#
# g = re.findall('abcadcabc', content1)
# d = re.findall('(abc)(adc)(abc){2}', content1)
# f = re.findall('(abc)(abc)(adc){2}', content2)
#
#
# print(g)
# print('{:-^20}'.format('d'))
# print(d)
# print('{:-^20}'.format('f'))
# print(f)

# a = 'EXCEL12233 dbad111c PPT233ababcdef'
# d = re.findall('[a-zA-Z]{3,5}', a)
# print(d)

# 3位区号，8位本地号（区号0开头）
# 4位区号，7位本地号（区号0开头）
# 区号可以用小括号括起来，也可以不用。区号与本地号间可以用连字号或空格间隔，也可以没有间隔。

# a = '''
# 010-12345678
# 0551-1234567
# (0551)1234567
# 0551 1234567
# 05511234567
# 01012345678
# (0551) 1234567
#
# 0551)1234567
# (0551))1234567
# 0551-12345678
# 12345678
# 1234567
# 12345678-12345
# '''
# e = re.findall(r'^\(0\d{2}\)[- ]?\d{8}|0\d{2}[- ]?\d{8}|\(0\d{3}\)[- ]?\d{7}|0\d{3}[- ]?\d{7}$', a, re.M)
# print(e)

# ^\(0\d{2}\)[- ]?\d{8}|0\d{2}[- ]?\d{8}$  05511234567  也可以取出即使是按照3/8取法，只要是11位就行
# ^\(0\d{3}\)[- ]?\d{7}|0\d{3}[- ]?\d{7}$  01012345678  也可以取出即使是按4/7取法，只要是11位就行

# ^(\(0\d{2,3}\)[- ]?\d{7,8}|0\d{2,3}[- ]?\d{7,8})$

# a = '''
# 不含字母的字符串
# Hello 你好
# HELLO WORLD
# 123數字
# name你
# name123_
# 你好吗?!@
# ?!@你好吗
# _123 你
# _123 name 12 name
# '''
#
# r = re.findall(r'\b[^A-Za-z\n\s]+\b', a, re.M)
# c = re.findall(r'\b[^A-Za-z\n\s]+|[?!@]+\b', a, re.M)
# d = re.findall(r'^[^A-Za-z]+$|\b[^A-Za-z\n\s]+\b', a, re.M)
#
# print(r)
# print(c)
# print(d)
#
import re

line = "Hello Hello Hello Hello"
match = re.search(r'(\b\w+\b)\1+', line)

