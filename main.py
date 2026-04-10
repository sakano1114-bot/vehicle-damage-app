#!/usr/bin/env python3
"""
車両損傷AI解析Webアプリ
FastAPI + Claude Vision API (ANTHROPIC_API_KEY 使用)
"""

import base64
import json
import os
from pathlib import Path

import anthropic
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse

# ============================================================
# 定数
# ============================================================

SUPPORTED_MEDIA_TYPES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".gif": "image/gif",
    ".webp": "image/webp",
}

SYSTEM_PROMPT = """あなたは自動車損傷の専門家です。
車両の損傷写真を分析し、損傷部位・損傷種別・修理方法を正確に評価します。
板金・塗装・部品交換の専門知識を持ち、作業時間の見積もりも行います。
回答は必ずJSON形式のみで出力してください。説明文や前置きは不要です。"""

ANALYSIS_PROMPT = """以下の車両損傷写真を詳細に解析してください。

損傷が確認できる全ての部位について評価し、以下のJSON形式のみで回答してください。
マークダウンのコードブロック記法（```）は使わず、JSONのみを出力してください。

{
  "damaged_parts": [
    {
      "part_name": "部位名（例：フロントバンパー、左フロントフェンダー、ボンネット等）",
      "damage_type": "損傷種別（凹み/傷/割れ/要交換 のいずれか）",
      "severity": "重大度（軽微/中程度/重大 のいずれか）",
      "repair_method": "修理方法（板金/塗装/部品交換 から該当するもの、複数の場合はカンマ区切り）",
      "estimated_work_hours": 推定作業時間（数値、単位：時間）,
      "confidence": 判定信頼度（0.0〜1.0の数値）
    }
  ],
  "total_estimated_hours": 全部位の合計推定作業時間（数値）,
  "notes": "全体的な損傷状況のコメント、注意点、追加情報など"
}

評価基準：
- 損傷種別：「凹み」= 変形あり・割れなし、「傷」= 表面の擦り傷・塗装剥がれ、「割れ」= 亀裂・破損、「要交換」= 修復不可能な損傷
- 重大度：「軽微」= 目立たない小さな損傷、「中程度」= 明らかな損傷だが機能に影響なし、「重大」= 大きな変形や機能への影響あり
- 作業時間：プロの板金・塗装技術者1名が必要とする時間（時間単位）
- 信頼度：画像の鮮明度・損傷の視認性・角度による判定可能性を考慮した0〜1の値"""


# ============================================================
# FastAPI アプリ
# ============================================================

app = FastAPI(title="車両損傷AI解析システム")


def get_client() -> anthropic.Anthropic:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="ANTHROPIC_API_KEY が環境変数に設定されていません",
        )
    return anthropic.Anthropic(api_key=api_key)


def parse_json_response(text: str) -> dict:
    """レスポンステキストからJSONを抽出してパースする"""
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0]
    elif "```" in text:
        text = text.split("```")[1].split("```")[0]
    else:
        start = text.find("{")
        end = text.rfind("}") + 1
        if start != -1 and end > start:
            text = text[start:end]
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError as e:
        raise ValueError(f"JSONの解析に失敗しました: {e}") from e


@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = Path(__file__).parent / "templates" / "index.html"
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))


@app.post("/analyze")
async def analyze(files: list[UploadFile] = File(...)):
    """
    アップロードされた車両画像を解析してJSON結果を返す。
    複数枚まとめて同一車両の複数アングルとして解析する。
    """
    if not files:
        raise HTTPException(status_code=400, detail="画像ファイルを選択してください")
    if len(files) > 10:
        raise HTTPException(status_code=400, detail="一度に解析できるのは最大10枚です")

    client = get_client()

    # メッセージコンテンツを構築
    content = []
    for i, file in enumerate(files, 1):
        suffix = Path(file.filename or "").suffix.lower()
        media_type = SUPPORTED_MEDIA_TYPES.get(suffix)
        if not media_type:
            raise HTTPException(
                status_code=400,
                detail=f"サポートされていない形式です: {file.filename} (対応: JPG/PNG/GIF/WEBP)",
            )

        image_bytes = await file.read()
        if len(image_bytes) > 20 * 1024 * 1024:  # 20MB limit
            raise HTTPException(
                status_code=400,
                detail=f"ファイルサイズが大きすぎます: {file.filename} (最大20MB)",
            )

        image_data = base64.standard_b64encode(image_bytes).decode("utf-8")
        content.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": media_type,
                "data": image_data,
            },
        })
        content.append({
            "type": "text",
            "text": f"[写真{i}/{len(files)}: {file.filename}]",
        })

    content.append({"type": "text", "text": ANALYSIS_PROMPT})

    try:
        response = client.messages.create(
            model="claude-opus-4-6",
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": content}],
        )
    except anthropic.APIStatusError as e:
        raise HTTPException(status_code=502, detail=f"Claude API エラー: {e.message}") from e
    except anthropic.APIConnectionError as e:
        raise HTTPException(status_code=502, detail=f"API接続エラー: {str(e)}") from e

    response_text = ""
    for block in response.content:
        if block.type == "text":
            response_text = block.text
            break

    if not response_text:
        raise HTTPException(status_code=502, detail="APIから有効なレスポンスが得られませんでした")

    try:
        result = parse_json_response(response_text)
    except ValueError as e:
        raise HTTPException(status_code=502, detail=str(e)) from e

    return result
