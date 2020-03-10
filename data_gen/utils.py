
# def trans_by_opencc(word):
# #     #将简体转换成繁体
# #     # cc = opencc.OpenCC('s2t')
# #     cc = opencc.OpenCC('mix2t')
# #     return cc.convert(word)
import langconv


def trans_by_zhtools(word):
    # 将简体转换成繁体
    word = langconv.Converter('zh-hant').convert(word)
    return word