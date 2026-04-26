from django.shortcuts import render
from django.views.generic import TemplateView
import os
import datetime
import gc

# Pytorch , transformers
from django.http import HttpResponse
# from transformers import AutoModelForQuestionAnswering, BertJapaneseTokenizer
from transformers import AutoModelForQuestionAnswering, AutoTokenizer
import torch

# CSRF
from django.views.decorators.csrf import csrf_exempt

model_name = 'KoichiYasuoka/bert-base-japanese-wikipedia-ud-head'
# model = AutoModelForQuestionAnswering.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(
    model_name,
    # メモリ消費量を減らすため、通常32bitで動くがモデルを16bitに落とす（精度は少し低下）
    # torch_dtype=torch.float16,
    low_cpu_mem_usage=True
)
model.eval()
gc.collect()
# tokenizer = BertJapaneseTokenizer.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

@csrf_exempt
# --- 画面表示 ---
# トップページの表示設定
def index(request):
    return render(request, 'index.html')

# --- BERT データの処理 ---
def reply(question):

    # コンテキスト設定
    # context = (
    #     "当配送サービスの営業時間は午前9時から午後6時です。 "
    #     "配送料は一律500円です。 "
    #     "領収書の発行はマイページから可能です。 "
    #     "再発行を希望される場合はカスタマーサポートへご連絡ください。 "
    #     "お届け日は注文から通常3日以内です。"
    # )

    # 知識ファイルのパス（chatbot_appの1つ上の階層にあるknowledge.txtを探す）
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'knowledge.txt')
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            full_context = f.read()
    except FileNotFoundError:
        return "回答が見つかりませんでした。"
    
    # 質問に関連する「行」だけを絞り込む（フィルタリング）
    sentences = full_context.splitlines()
    relevant_sentences = []

    keywords = ["営業", "送料", "配送料", "領収", "請求", "届", "いつ", "発送", "キャンセル", "返品", "支払い", "支払", "指定", "変更", "交換", "傷み", "傷", "腐", "不良"]
    # 核心キーワードが見つかったら、それを target_keywords に入れる
    # ユーザーが「お届けについて」と言っても、「届け」がヒットする
    target_keywords = [w for w in keywords if w in question]

    # もしキーワードが含まれていたら、そのキーワードを含む行だけをピックアップ
    if target_keywords:
        for s in sentences:
            if any(k in s for k in target_keywords):
                relevant_sentences.append(s)

    # もし絞り込めなかったり、絞り込みが少なすぎたら500文字目までを対象にする（ただし512制限のため上位数行）
    if relevant_sentences:
        context = "\n".join(relevant_sentences)
    else:
        context = full_context[:500] # 最大文字数(512)対策
    
    # --- AIへのヒントを強化して渡す ---
    q_logic = question

    # 営業関連
    if any(word in question for word in ["営業", "時間", "何時", "休み"]):
        q_logic = "営業時間は何時から何時までですか？"
    # 送料関連
    elif any(word in question for word in ["配送", "送料", "無料", "送料無料", "手数料", "運賃", "クール"]):
        q_logic = "配送料はいくらですか？送料無料になりますか？"
    # 領収書・請求書関連
    elif any(word in question for word in ["領収", "発行", "レシート", "インボイス"]):
        q_logic = "領収書の発行はできますか？"
    # お届け・発送関連
    elif any(word in question for word in ["いつ", "届く", "届かない", "予定", "到着"]):
        q_logic = "お届け予定日いつですか。商品はいつ届きますか？"
        relevant_sentences = [s for s in sentences if "お届け予定日について" in s or "通常3日以内" in s]
    elif any(word in question for word in ["指定", "変更", "日にち", "日付"]):
        q_logic = "お届け日の指定や変更は可能ですか？"
        relevant_sentences = [s for s in sentences if "指定" in s or "変更" in s]
    # 支払い関連
    elif any(word in question for word in ["支払い", "支払", "決済", "カード", "ペイペイ", "代金"]):
        q_logic = "支払方法は何がありますか？"
    # キャンセル関連
    elif any(word in question for word in ["キャンセル", "変更",]):
        q_logic = "注文のキャンセルや変更はできますか？"
    # 不良品    
    elif any(word in question for word in ["返金", "返品", "傷み", "傷んで", "腐っ", "交換", "不良"]):
        q_logic = "不良品や傷みがあった場合、返品・交換はできますか？" # += ではなく = で上書き
    
    if relevant_sentences:
        context = "\n".join(relevant_sentences)[:500]
    
    # BERT処理
    inputs = tokenizer.encode_plus(
        q_logic, 
        context, 
        add_special_tokens=True, 
        return_tensors="pt", 
        truncation=True,    # BERTの最大値512を超えたら切り捨てる設定
        max_length=512      # 最大長を指定
    )
    input_ids = inputs["input_ids"].tolist()[0]
    
    with torch.no_grad():
        output = model(**inputs)

    # 自信度（スコア）の取得
    start_logits = output.start_logits
    confidence_score = float(torch.max(start_logits))
    print(f"DEBUG SCORE: {confidence_score}")

    # 信頼度チェック１
    # 関連するキーワードがない場合
    if confidence_score < 0.5:
        return "申し訳ございません、ご質問に関する回答が見つかりませんでした。<br>オペレーターより回答をご希望の場合は、以下のお問い合わせ窓口までご連絡ください。<br><br>＜電話窓口＞<br>info@○○○○.ne.jp<br>TEL：0120-000-0000<br>営業時間：9:00～17:00", confidence_score
    
    answer_start = torch.argmax(output.start_logits)
    answer_end = torch.argmax(output.end_logits) + 1

    # BERTが選んだスパンをテキスト化
    raw_answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end])).replace(' ', '')

    # 信頼度チェック２
    # 抜き出した範囲がBERTの特殊な記号（文章の先頭など）を答えとして選んでしまった場合
    if not raw_answer or "[CLS]" in raw_answer:
        return "申し訳ございません、回答が見つかりませんでした。<br>オペレーターより回答をご希望の場合は、以下のお問い合わせ窓口までご連絡ください。<br><br>＜電話窓口＞<br>info@○○○○.ne.jp<br>TEL：0120-000-0000<br>営業時間：9:00～17:00", confidence_score
    

    sentences = context.splitlines() 

    # --- 第1段階：BERTが抜き出した単語(2文字以上)で「行」を探す
    for sentence in sentences:
        # 抜き出した答え（例：9:00～17:00）がその行に含まれているか
        # または、質問のキーワードが含まれているか
        if len(raw_answer) > 1 and raw_answer in sentence:
            return sentence.strip(), confidence_score
        
    # --- 第2段階：もし行が見つからなかったら、予備のキーワードで再検索
    for sentence in sentences:
        if any(word in q_logic and word in sentence for word in ["営業", "送料", "領収", "届", "到着", "支払い", "キャンセル", "返品", "交換", "腐", "傷", "不良", "指定", "変更"]):
            return sentence.strip(), confidence_score

    final_answer = raw_answer
    return final_answer, confidence_score

# --- JavaScriptからのリクエストを受け取る
def bot_response(request):

    input_data = request.POST.get('input_text')
    if not input_data:
        return HttpResponse('<h2>一時的なエラー</h2>質問内容が入力されていません。質問内容を入力後、送信ボタンを押してください。<br>最初からやり直してください。', status=400)

    # ターミナルへのデバッグ表示
    print(f"DEBUG: POST data is {request.POST}")

    # chat log
    # 1. 回答とスコアの取得(AIに回答を依頼)
    bot_answer, score = reply(input_data)

    # 2. 問い合わせ時刻を取得
    now_obj = datetime.datetime.now()
    now_str = now_obj.strftime('%Y-%m-%d %H:%M:%S')    

    # 3. ログ内容を作成 (①日時・質問、②スコア・回答)
    log_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'chat_log.txt')
    
    # HTMLタグを除去して保存用に整形
    clean_answer = bot_answer.replace('<br>', ' ').replace('\n', ' ')

    log_entry = (
        f"--- Chat Log ---\n"
        f"①日時: {now_str}\n"
        f"  質問: {input_data}\n"
        f"②スコア: {score:.4f}\n"
        f"  回答: {clean_answer[:100]}...\n" # ログが長くなりすぎないよう100文字でカット
        f"-----------------\n\n"
    )

    # 4. ファイルに追記
    try:
        with open(log_file_path, 'a', encoding='utf-8') as f:
            f.write(log_entry)
            print(log_entry)
    except Exception as e:
        print(f"LOG ERROR: {e}")

    # 5. 画面に回答を返す
    return HttpResponse(f"{bot_answer}")