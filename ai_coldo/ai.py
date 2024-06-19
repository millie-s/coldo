import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
import MeCab

# MeCabのインスタンス作成
mecab = MeCab.Tagger("-Owakati")

# サンプルデータの作成
data = {
    'message': [
        '好きです！',
        'ブサイク',
        'ぶりっことかきもい',
        '私の住所は東京都世田谷区岡本１丁目１０−１です',
        '夢に向かって突っ走る姿、カッコいいです！',
        'ライブ見てから応援してます',
        '大好きです！',
        '私とか俺とか一人称コロコロ変わるのキツ',
        'パフォーマンスに感動しました',
        '出会えてよかった～！',
        'たくさんの良いことが訪れるように願っています～！',
        'いつも応援してます',
        'いい子ちゃんぶってるのやめたら',
        '活動応援してます',
        '電話番号教えて'
    ],
    'label': [ 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1]  # 0: Positive, 1: Negative
}

df = pd.DataFrame(data)

# データセットの分割
X_train, X_test, y_train, y_test = train_test_split(df['message'], df['label'], test_size=0.2, random_state=42)

# カスタムトークナイザ
def tokenizer(text):
    return mecab.parse(text).strip().split()

# パイプラインの作成
model = make_pipeline(TfidfVectorizer(tokenizer=tokenizer), LinearSVC())

# モデルの訓練
model.fit(X_train, y_train)

# モデルの評価（任意）
print("Test accuracy: {:.2f}".format(model.score(X_test, y_test)))

def replace_negative_expressions(message, model):
    # メッセージをMeCabで単語に分割
    words = tokenizer(message)
    negative_words = []

    # ネガティブな単語をxxxに置換
    for i, word in enumerate(words):
        if model.predict([word])[0] == 1:
            negative_words.append(word)
            words[i] = '♦♦♦'

    # 置換後のメッセージを再結合
    filtered_message = ' '.join(words)

    return filtered_message, negative_words

def generate_positive_reply(input_message, model):
    # Mapping of negative phrases to positive responses
    positive_responses = {
        "まじきもいから活動やめたら": "君のユニークな個性は他の場所でもっと輝くと思うから、別の活動にチャレンジしてみたらどうかな？",
        "いつも応援してます": "いつも応援してます。",
        "ぶりっこきつ":"もっと自然体の君が見たいな！",
        "今日のテレビでの笑顔ぎこちなかったよ？":"今日のテレビの笑顔も良いけど昨日の方が好きだったな",
        "引退したら良いと思う":"新しいチャレンジを探してみたらどうかな？"
    }

    # Look up positive response based on input_message
    if input_message in positive_responses:
        return positive_responses[input_message]
    else:
        return "すみません、そのメッセージに対するポジティブな返答は用意できませんでした。他のメッセージをお試しください。"

# Main function
def main():
    flag = 0
    while flag == 0:
        input_message = input("メッセージを入力してください (終了するには 'exit' と入力): ")
        if input_message.lower() == "exit":
            flag = 1
        else:
            # ネガティブな表現をフィルタリング
            filtered_message, negative_words = replace_negative_expressions(input_message, model)

            # ポジティブな返答生成
            positive_reply = generate_positive_reply(filtered_message, model)

            print("フィルタリングされたメッセージ:", filtered_message)
            #print("ネガティブと認識された単語:", negative_words)
            #print("ポジティブな返答:", positive_reply)

if __name__ == "__main__":
    main()
