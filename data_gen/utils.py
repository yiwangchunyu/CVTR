
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

if __name__=='__main__':
    print(trans_by_zhtools("中央国务院关于推动形成全面开放新格局决策部署完善综合保税区营商环境进一步促进贸易投资便利化有利于稳外贸稳外资保持合理进出口规模打造对外开放新高地"))