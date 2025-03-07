from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import Application, CallbackQueryHandler, CommandHandler, ContextTypes
import ccxt
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import asyncio
import json
import pytz
import re
import os
import plotly.figure_factory as ff
import numpy as np
import requests

# Token bot từ BotFather
TOKEN = "8117660223:AAHCEFYmz7GxYd9O7H-wIsp2P-J_o-kxP7s"

# Địa chỉ Webhook (thay YOUR_RENDER_URL bằng URL ứng dụng Render của bạn)
WEBHOOK_URL = f"https://telegrambot-an3l.onrender.com"
# Khởi tạo KuCoin
exchange = ccxt.kucoin()
# Lưu trữ lịch sử tín hiệu
signal_history = {}

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Gửi tin nhắn chào mừng và hướng dẫn."""
    await update.message.reply_text(
        "Chào mừng! Tôi là bot hỗ trợ cảnh báo tín hiệu mua/bán tiền mã hóa.\n"
        "Dưới đây là các lệnh bạn có thể sử dụng:\n"
        "Gõ /chart <mã giao dịch> để xem biểu đồ kỹ thuật (ví dụ: /chart BTC/USDT).\n"
        "Gõ /top để xem top 10 cặp giao dịch tăng, giảm mạnh nhất 24 giờ qua.\n"
        "Gõ /signal <mã giao dịch> để xem lịch sử tín hiệu mua bán trong 7 ngày qua.\n"
        "Gõ /smarttrade <mã giao dịch> để xem thông tin và tín hiệu mua bán mới nhất.\n"
        "Gõ /list để xem top 10 cặp giao dịch có tín hiệu mua bán gần đây.\n"
        "Gõ /info để xem thông tin đồng coin.\n"
        "Gõ /heatmap để xem heatmap của 100 đồng coin.\n"
        "Gõ /desc để xem mô tả đồng coin."
    )


# Khởi tạo múi giờ Việt Nam
vietnam_tz = pytz.timezone('Asia/Ho_Chi_Minh')


def escape_markdown(text: str, ignore: list = None) -> str:
    """
    Thoát các ký tự đặc biệt cho Markdown v2.
    Các ký tự trong danh sách `ignore` sẽ không bị thoát.
    """
    if ignore is None:
        ignore = []
    # Các ký tự Markdown cần thoát
    escape_chars = r"_*[]()~`>#+-=|{}.!"
    # Loại bỏ các ký tự trong danh sách ignore
    for char in ignore:
        escape_chars = escape_chars.replace(char, "")
    # Thay thế các ký tự cần thoát bằng cách thêm dấu '\'
    return re.sub(f"([{re.escape(escape_chars)}])", r"\\\1", text)


async def current_price(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    try:
        symbol = context.args[0] if context.args else None
        if not symbol:
            await update.message.reply_text("Vui lòng cung cấp mã giao dịch. Ví dụ: /smarttrade BTC/USDT")
            return

        markets = exchange.load_markets()
        if symbol not in markets:
            await update.message.reply_text(f"Mã giao dịch không hợp lệ: {symbol}. Vui lòng kiểm tra lại.")
            return

        quote_currency = symbol.split('/')[1]
        ticker = exchange.fetch_ticker(symbol)
        current_price = ticker['last']
        percentage_change = ticker['percentage']
        volume_24h = ticker.get('quoteVolume', 0)

        timestamp = (
            pd.to_datetime(ticker['timestamp'], unit='ms')
            .tz_localize('UTC')
            .tz_convert(vietnam_tz)
            .strftime('%Y-%m-%d %H:%M:%S')
        )

        timeframe = '2h'
        limit = 500
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = (
            pd.to_datetime(df['timestamp'], unit='ms')
            .dt.tz_localize('UTC')
            .dt.tz_convert(vietnam_tz)
        )

        if len(df) < 100:
            await update.message.reply_text("Không đủ dữ liệu để tính toán chỉ báo kỹ thuật. Vui lòng thử lại sau.")
            return

        df['MA50'] = df['close'].rolling(window=50).mean()
        df['MA100'] = df['close'].rolling(window=100).mean()
        df['EMA12'] = df['close'].ewm(span=12).mean()
        df['EMA26'] = df['close'].ewm(span=26).mean()
        df['MACD'] = df['EMA12'] - df['EMA26']
        df['Signal'] = df['MACD'].ewm(span=9).mean()
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        df['BB_Middle'] = df['close'].rolling(window=20).mean()
        df['BB_Upper'] = df['BB_Middle'] + 2 * df['close'].rolling(window=20).std()
        df['BB_Lower'] = df['BB_Middle'] - 2 * df['close'].rolling(window=20).std()

        trend = "Không xác định"
        if len(df) > 1:
            last_row = df.iloc[-1]
            prev_row = df.iloc[-2]
            if last_row['close'] > last_row['MA50'] and last_row['close'] > last_row['MA100'] and last_row['MA50'] > prev_row['MA50']:
                trend = "TĂNG"
            elif last_row['close'] < last_row['MA50'] and last_row['close'] < last_row['MA100'] and last_row['MA50'] < prev_row['MA50']:
                trend = "GIẢM"
            else:
                trend = "ĐI NGANG"

        signals = []
        for _, row in df.iterrows():
            if row['close'] > row['MA50'] and row['MACD'] > row['Signal'] and row['RSI'] < 30:
                signals.append({"type": "MUA", "price": row['close'], "timestamp": row['timestamp']})
            elif row['close'] <= row['BB_Lower']:
                signals.append({"type": "MUA", "price": row['close'], "timestamp": row['timestamp']})
            elif row['close'] < row['MA50'] and row['MACD'] < row['Signal'] and row['RSI'] > 70:
                signals.append({"type": "BÁN", "price": row['close'], "timestamp": row['timestamp']})
            elif row['close'] >= row['BB_Upper']:
                signals.append({"type": "BÁN", "price": row['close'], "timestamp": row['timestamp']})

        recent_signal = signals[-1] if signals else None
        position_info = "Không có tín hiệu mua/bán gần đây."

        if recent_signal:
            signal_age = (pd.Timestamp.utcnow().tz_convert(vietnam_tz) - recent_signal['timestamp']).total_seconds() / 3600
            position_status = "THEO DÕI" if signal_age > 2 else recent_signal['type']
            if recent_signal['type'] == "MUA":
                profit_loss = ((current_price - recent_signal['price']) / recent_signal['price']) * 100
                profit_color = (
                    f"{profit_loss:.2f}% 🟢" if profit_loss > 0 else
                    f"{profit_loss:.2f}% 🔴" if profit_loss < 0 else
                    f"{profit_loss:.2f}% 🟡"
                )
                position_info = (
                    f"- Xu hướng: **{trend}**\n"
                    f"- Vị thế hiện tại: **{position_status}**\n"
                    f"- Ngày mua: {recent_signal['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}\n"
                    f"- Giá mua: {recent_signal['price']:.2f} {quote_currency}\n"
                    f"- Lãi/Lỗ: {profit_color}"
                )
            elif recent_signal['type'] == "BÁN":
                buy_signals = [s for s in signals if s['type'] == "MUA" and s['timestamp'] < recent_signal['timestamp']]
                if buy_signals:
                    prior_buy = max(buy_signals, key=lambda x: x['timestamp'])  # Chọn lần mua gần nhất
                    profit_loss = ((recent_signal['price'] - prior_buy['price']) / prior_buy['price']) * 100
                    profit_color = (
                        f"{profit_loss:.2f}% 🟢" if profit_loss > 0 else
                        f"{profit_loss:.2f}% 🔴" if profit_loss < 0 else
                        f"{profit_loss:.2f}% 🟡"
                    )
                    position_info = (
                        f"- Xu hướng: **{trend}**\n"
                        f"- Vị thế hiện tại: **{position_status}**\n"
                        f"- Ngày mua: {prior_buy['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}\n"
                        f"- Giá mua: {prior_buy['price']:.2f} {quote_currency}\n"
                        f"- Ngày bán: {recent_signal['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}\n"
                        f"- Giá bán: {recent_signal['price']:.2f} {quote_currency}\n"
                        f"- Lãi/Lỗ: {profit_color}"
                    )
                else:
                    position_info = (
                        f"- Xu hướng: **{trend}**\n"
                        f"- Vị thế hiện tại: **{position_status}**\n"
                        f"- Ngày bán: {recent_signal['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}\n"
                        f"- Giá bán: {recent_signal['price']:.2f} {quote_currency}\n"
                        f"- Lãi/Lỗ: Không xác định (không có tín hiệu mua trước đó)."
                    )


        message = escape_markdown(
            f"Thông tin giá hiện tại cho {symbol}:\n"
            f"- Giá hiện tại: {current_price:.2f} {quote_currency}\n"
            f"- Biến động trong 24 giờ qua: {percentage_change:.2f}%\n"
            f"- Khối lượng giao dịch trong 24 giờ qua: {volume_24h:.2f} {quote_currency}\n"
            f"- Thời gian cập nhật: {timestamp}\n\n"
            f"Thông tin vị thế:\n{position_info}",
            ignore=["*"]
        )
        await update.message.reply_text(message, parse_mode="MarkdownV2")

    except Exception as e:
        await update.message.reply_text(f"Đã xảy ra lỗi: {e}")

async def chart(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Tạo và gửi biểu đồ kỹ thuật."""
    try:
        symbol = context.args[0] if context.args else context.chat_data.get("symbol")
        if not symbol:
            await update.message.reply_text("Vui lòng cung cấp mã giao dịch. Ví dụ: /chart BTC/USDT")
            return

        timeframe = '1h'
        limit = 8760

        markets = exchange.load_markets()
        if symbol not in markets:
            await update.message.reply_text(f"Mã giao dịch không hợp lệ: {symbol}. Vui lòng kiểm tra lại.")
            return

        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        # Chuyển đổi timestamp sang giờ Việt Nam
        df['timestamp'] = (
            pd.to_datetime(df['timestamp'], unit='ms')
            .dt.tz_localize('UTC')
            .dt.tz_convert(vietnam_tz)
        )

        # Tính toán các chỉ báo kỹ thuật
        df['MA50'] = df['close'].rolling(window=50).mean()
        df['MA100'] = df['close'].rolling(window=100).mean()

        # Bollinger Bands
        df['BB_Middle'] = df['close'].rolling(window=20).mean()
        df['BB_Upper'] = df['BB_Middle'] + 2 * df['close'].rolling(window=20).std()
        df['BB_Lower'] = df['BB_Middle'] - 2 * df['close'].rolling(window=20).std()

        # MACD
        df['EMA12'] = df['close'].ewm(span=12).mean()
        df['EMA26'] = df['close'].ewm(span=26).mean()
        df['MACD'] = df['EMA12'] - df['EMA26']
        df['Signal'] = df['MACD'].ewm(span=9).mean()
        df['MACD_Hist'] = df['MACD'] - df['Signal']

        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))

        # Biểu đồ Candlestick và MACD được đặt riêng biệt
        fig = make_subplots(
            rows=4,  # Tăng số lượng hàng lên 4 để tách MACD khỏi biểu đồ giá
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.5, 0.2, 0.2, 0.1],  # Cập nhật chiều cao từng hàng
            specs=[[{"secondary_y": True}], [{}], [{}], [{}]]
        )

        # Candlestick và Bollinger Bands
        fig.add_trace(go.Candlestick(
            x=df['timestamp'],
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name="Candlestick"
        ), row=1, col=1, secondary_y=False)

        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['BB_Upper'],
            mode='lines',
            line=dict(color='red', width=1),
            name='BB Upper'
        ), row=1, col=1, secondary_y=False)

        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['BB_Middle'],
            mode='lines',
            line=dict(color='blue', width=1),
            name='BB Middle'
        ), row=1, col=1, secondary_y=False)

        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['BB_Lower'],
            mode='lines',
            line=dict(color='green', width=1),
            name='BB Lower'
        ), row=1, col=1, secondary_y=False)

        # Thêm các đường MA50 và MA100
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['MA50'],
            mode='lines',
            line=dict(color='orange', width=1.5),
            name='MA50'
        ), row=1, col=1, secondary_y=False)

        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['MA100'],
            mode='lines',
            line=dict(color='purple', width=1.5),
            name='MA100'
        ), row=1, col=1, secondary_y=False)

        # Biểu đồ khối lượng bên trục y2, cùng màu với giá
        volume_colors = [
            'green' if row['close'] > row['open'] else 'red'
            for _, row in df.iterrows()
        ]
        fig.add_trace(go.Bar(
            x=df['timestamp'],
            y=df['volume'],
            name='Volume',
            marker_color=volume_colors
        ), row=1, col=1, secondary_y=True)

        # Biểu đồ MACD (được chuyển sang hàng 2)
        fig.add_trace(go.Bar(
            x=df['timestamp'],
            y=df['MACD_Hist'],
            name='MACD Histogram'
        ), row=2, col=1)

        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['MACD'],
            mode='lines',
            line=dict(color='green', width=1),
            name='MACD'
        ), row=2, col=1)

        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['Signal'],
            mode='lines',
            line=dict(color='red', width=1),
            name='Signal'
        ), row=2, col=1)

        # Biểu đồ RSI (hàng 3)
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['RSI'],
            mode='lines',
            line=dict(color='purple', width=1),
            name='RSI'
        ), row=3, col=1)

        # Đường giới hạn RSI
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=[70] * len(df),
            mode='lines',
            line=dict(color='red', dash='dash'),
            name='Overbought (70)'
        ), row=3, col=1)

        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=[30] * len(df),
            mode='lines',
            line=dict(color='blue', dash='dash'),
            name='Oversold (30)'
        ), row=3, col=1)

        # Layout
        fig.update_layout(
            title=f"BIỂU ĐỒ PHÂN TÍCH KỸ THUẬT (1H) CỦA {symbol}",
            template="plotly_dark",
            height=1200,  # Tăng chiều cao biểu đồ tổng thể
            xaxis_rangeslider_visible=False
        )

        # Lưu biểu đồ thành HTML
        temp_file = f"{symbol.replace('/', '_')}_chart.html"
        fig.write_html(temp_file)

        # Gửi file HTML qua Telegram
        if update.callback_query:
            with open(temp_file, 'rb') as html_file:
                await update.callback_query.message.reply_document(document=html_file, filename=temp_file)
        else:
            with open(temp_file, 'rb') as html_file:
                await update.message.reply_document(document=html_file, filename=temp_file)

        # Xóa file tạm
        os.remove(temp_file)
    except Exception as e:
        if update.callback_query:
            await update.callback_query.message.reply_text(f"Đã xảy ra lỗi: {e}")
        else:
            await update.message.reply_text(f"Đã xảy ra lỗi: {e}")



async def top(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Gửi danh sách top 10 cặp giao dịch tăng, giảm mạnh nhất với nút tương tác."""
    try:
        # Lấy dữ liệu thị trường từ KuCoin
        markets = exchange.fetch_tickers()
        data = []

        # Tính toán phần trăm biến động giá và khối lượng giao dịch
        for symbol, ticker in markets.items():
            change = ticker.get('percentage')
            if change is not None:
                data.append((symbol, change))


        # Lấy top 10 tăng, giảm mạnh nhất 
        top_gainers = sorted(data, key=lambda x: x[1], reverse=True)[:10]
        top_losers = sorted(data, key=lambda x: x[1])[:10]

        # Tạo danh sách nút tương tác cho top tăng
        gainers_keyboard = [
            [InlineKeyboardButton(f"{symbol}: +{change:.2f}%", callback_data=symbol)]
            for symbol, change in top_gainers
        ]

        # Tạo danh sách nút tương tác cho top giảm
        losers_keyboard = [
            [InlineKeyboardButton(f"{symbol}: {change:.2f}%", callback_data=symbol)]
            for symbol, change in top_losers
        ]


        # Gửi danh sách top tăng mạnh nhất
        await update.message.reply_text(
            "Top 10 cặp giao dịch tăng mạnh nhất trong 24 giờ qua:",
            reply_markup=InlineKeyboardMarkup(gainers_keyboard)
        )

        # Gửi danh sách top giảm mạnh nhất
        await update.message.reply_text(
            "Top 10 cặp giao dịch giảm mạnh nhất trong 24 giờ qua:",
            reply_markup=InlineKeyboardMarkup(losers_keyboard)
        )


    except Exception as e:
        await update.message.reply_text(f"Đã xảy ra lỗi: {e}")



async def button(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Xử lý nút bấm từ danh sách /top để hiển thị biểu đồ kỹ thuật."""
    query = update.callback_query
    await query.answer()


    # Lấy mã giao dịch từ callback_data
    symbol = query.data
    context.chat_data["symbol"] = symbol  # Lưu vào chat_data để gọi lại nếu cần
    await chart(update, context)


async def list_signals(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Hiển thị top 10 cặp giao dịch có tín hiệu mua và tín hiệu bán gần đây."""
    try:
        # Lấy danh sách mã giao dịch
        markets = exchange.load_markets()
        symbols = list(markets.keys())
        timeframe = '2h'
        limit = 200
        buy_signals = []
        sell_signals = []

        for symbol in symbols:
            try:
                # Lấy dữ liệu từ KuCoin
                ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
                df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

                # Tính toán các chỉ báo kỹ thuật
                df['MA50'] = df['close'].rolling(window=50).mean()
                df['EMA12'] = df['close'].ewm(span=12).mean()
                df['EMA26'] = df['close'].ewm(span=26).mean()
                df['MACD'] = df['EMA12'] - df['EMA26']
                df['Signal'] = df['MACD'].ewm(span=9).mean()
                delta = df['close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                df['RSI'] = 100 - (100 / (1 + rs))
                df['BB_Middle'] = df['close'].rolling(window=20).mean()
                df['BB_Upper'] = df['BB_Middle'] + 2 * df['close'].rolling(window=20).std()
                df['BB_Lower'] = df['BB_Middle'] - 2 * df['close'].rolling(window=20).std()

                # Lấy tín hiệu gần nhất
                last_row = df.iloc[-1]
                current_time = last_row['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
                current_price = last_row['close']

                # Lấy đơn vị giá từ cặp giao dịch
                quote_currency = symbol.split('/')[1] if '/' in symbol else 'USD'

                # Tín hiệu mua
                if last_row['close'] > last_row['MA50'] and last_row['MACD'] > last_row['Signal'] and last_row['RSI'] < 30:
                    buy_signals.append((symbol, current_price, current_time, quote_currency))
                elif last_row['close'] <= last_row['BB_Lower']:
                    buy_signals.append((symbol, current_price, current_time, quote_currency))

                # Tín hiệu bán
                if last_row['close'] < last_row['MA50'] and last_row['MACD'] < last_row['Signal'] and last_row['RSI'] > 70:
                    sell_signals.append((symbol, current_price, current_time, quote_currency))
                elif last_row['close'] >= last_row['BB_Upper']:
                    sell_signals.append((symbol, current_price, current_time, quote_currency))

            except Exception as e:
                print(f"Lỗi khi xử lý {symbol}: {e}")
                continue

        # Lấy top 10 tín hiệu mua và bán
        top_buy_signals = sorted(buy_signals, key=lambda x: x[2], reverse=True)[:10]
        top_sell_signals = sorted(sell_signals, key=lambda x: x[2], reverse=True)[:10]

        # Tạo danh sách nút tương tác cho tín hiệu mua
        buy_keyboard = [
            [InlineKeyboardButton(f"{symbol}: Mua ({price:.8f} {unit})", callback_data=symbol)]
            for symbol, price, _, unit in top_buy_signals
        ]

        # Tạo danh sách nút tương tác cho tín hiệu bán
        sell_keyboard = [
            [InlineKeyboardButton(f"{symbol}: Bán ({price:.8f} {unit})", callback_data=symbol)]
            for symbol, price, _, unit in top_sell_signals
        ]

        # Gửi danh sách tín hiệu mua
        if buy_keyboard:
            await update.message.reply_text(
                "Top 10 cặp giao dịch có tín hiệu MUA gần đây:",
                reply_markup=InlineKeyboardMarkup(buy_keyboard)
            )
        else:
            await update.message.reply_text("Hiện không có tín hiệu MUA nào gần đây.")

        # Gửi danh sách tín hiệu bán
        if sell_keyboard:
            await update.message.reply_text(
                "Top 10 cặp giao dịch có tín hiệu BÁN gần đây:",
                reply_markup=InlineKeyboardMarkup(sell_keyboard)
            )
        else:
            await update.message.reply_text("Hiện không có tín hiệu BÁN nào gần đây.")

    except Exception as e:
        await update.message.reply_text(f"Đã xảy ra lỗi: {e}")


async def signal(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Phân tích và gửi tín hiệu mua bán."""
    try:
        symbol = context.args[0] if context.args else None
        if not symbol:
            await update.message.reply_text("Vui lòng cung cấp mã giao dịch. Ví dụ: /signal BTC/USDT")
            return

        # Xác định đơn vị giá từ cặp giao dịch
        if "/" in symbol:
            base, quote = symbol.split("/")
            unit = quote
        else:
            await update.message.reply_text("Cặp giao dịch không hợp lệ. Vui lòng sử dụng định dạng như BTC/USDT.")
            return

        timeframe = '2h'
        limit = 500

        markets = exchange.load_markets()
        if symbol not in markets:
            await update.message.reply_text(f"Mã giao dịch không hợp lệ: {symbol}. Vui lòng kiểm tra lại.")
            return

        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        # Chuyển đổi timestamp sang giờ Việt Nam
        df['timestamp'] = (
            pd.to_datetime(df['timestamp'], unit='ms')
            .dt.tz_localize('UTC')
            .dt.tz_convert(vietnam_tz)
        )

        # Tính toán các chỉ báo kỹ thuật
        df['MA50'] = df['close'].rolling(window=50).mean()
        df['MA100'] = df['close'].rolling(window=100).mean()
        df['EMA12'] = df['close'].ewm(span=12).mean()
        df['EMA26'] = df['close'].ewm(span=26).mean()
        df['MACD'] = df['EMA12'] - df['EMA26']
        df['Signal'] = df['MACD'].ewm(span=9).mean()
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        df['BB_Middle'] = df['close'].rolling(window=20).mean()
        df['BB_Upper'] = df['BB_Middle'] + 2 * df['close'].rolling(window=20).std()
        df['BB_Lower'] = df['BB_Middle'] - 2 * df['close'].rolling(window=20).std()

        # Phát hiện tín hiệu mua bán hiện tại
        last_row = df.iloc[-1]  # Lấy dòng dữ liệu cuối cùng
        signals_now = []
        last_buy_price = None
        last_buy_price_global = None  # Lưu giá mua gần nhất hợp lệ

        # Thời điểm và giá hiện tại
        current_time = last_row['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
        current_price = last_row['close']

        # Phát hiện tín hiệu mua bán trong 7 ngày qua
        signals_past = []
        now = pd.Timestamp.now(tz=vietnam_tz)
        for index, row in df.iterrows():
            if row['timestamp'] < (now - pd.Timedelta(days=7)):
                continue

            # Tín hiệu mua trong 7 ngày qua
            if row['close'] > row['MA50'] and row['MACD'] > row['Signal'] and row['RSI'] < 30:
                last_buy_price_global = row['close']  # Cập nhật giá mua toàn cục
                profit_loss = ((current_price - last_buy_price_global) / last_buy_price_global) * 100
                profit_icon = "\U0001F7E2" if profit_loss >= 0 else "\U0001F534"
                signals_past.append(f"\U0001F7E2 Mua: Giá {row['close']:.2f} {unit} vào lúc {row['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}. {profit_icon} Lãi/Lỗ: {profit_loss:.2f}%")
            elif row['close'] <= row['BB_Lower']:
                last_buy_price_global = row['close']  # Cập nhật giá mua toàn cục
                profit_loss = ((current_price - last_buy_price_global) / last_buy_price_global) * 100
                profit_icon = "\U0001F7E2" if profit_loss >= 0 else "\U0001F534"
                signals_past.append(f"\U0001F7E2 Mua: Giá {row['close']:.2f} {unit} vào lúc {row['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}. {profit_icon} Lãi/Lỗ: {profit_loss:.2f}%")

            # Tín hiệu bán trong 7 ngày qua
            if row['close'] < row['MA50'] and row['MACD'] < row['Signal'] and row['RSI'] > 70:
                if last_buy_price_global is not None:  # Sử dụng giá mua toàn cục
                    profit_loss = ((row['close'] - last_buy_price_global) / last_buy_price_global) * 100
                    profit_icon = "\U0001F7E2" if profit_loss >= 0 else "\U0001F534"
                    signals_past.append(f"\U0001F534 Bán: Giá {row['close']:.2f} {unit} vào lúc {row['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}. {profit_icon} Lãi/Lỗ: {profit_loss:.2f}%")
            elif row['close'] >= row['BB_Upper']:
                if last_buy_price_global is not None:  # Sử dụng giá mua toàn cục
                    profit_loss = ((row['close'] - last_buy_price_global) / last_buy_price_global) * 100
                    profit_icon = "\U0001F7E2" if profit_loss >= 0 else "\U0001F534"
                    signals_past.append(f"\U0001F534 Bán: Giá {row['close']:.2f} {unit} vào lúc {row['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}. {profit_icon} Lãi/Lỗ: {profit_loss:.2f}%")

        # Gửi tín hiệu qua Telegram
        signal_message = f"Tín hiệu giao dịch cho {symbol}:"

        if signals_past:
            signal_message += "\n\nTín hiệu trong 7 ngày qua:\n" + "\n".join(signals_past)
        else:
            signal_message += "\n\nKhông có tín hiệu trong 7 ngày qua."

        await update.message.reply_text(signal_message)

    except Exception as e:
        await update.message.reply_text(f"Đã xảy ra lỗi: {e}")


async def info(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Lấy thông tin chi tiết về một đồng coin từ CoinGecko dựa trên tên đầy đủ."""
    try:
        if not context.args:
            await update.message.reply_text("Vui lòng cung cấp tên coin. Ví dụ: /info bitcoin")
            return

        coin_name = "-".join(context.args).lower()  # Xử lý tên có dấu cách (ví dụ: "bitcoin cash" -> "bitcoin-cash")

        # Gọi API để lấy thông tin chi tiết của coin
        url = f"https://api.coingecko.com/api/v3/coins/{coin_name}"
        response = requests.get(url)
        if response.status_code != 200:
            await update.message.reply_text(f"Không tìm thấy thông tin về đồng coin: {coin_name}. Vui lòng kiểm tra lại.")
            return

        data = response.json()

        # Kiểm tra và xử lý NoneType trước khi format
        def safe_format(value, format_str="{:.2f}"):
            return format_str.format(value) if value is not None else "N/A"

        price_usd = safe_format(data['market_data']['current_price'].get('usd'))
        high_24h = safe_format(data['market_data']['high_24h'].get('usd'))
        all_time_high = safe_format(data['market_data']['ath'].get('usd'))  # Giá cao nhất từ khi niêm yết
        change_1h = safe_format(data['market_data']['price_change_percentage_1h_in_currency'].get('usd'))
        change_24h = safe_format(data['market_data']['price_change_percentage_24h_in_currency'].get('usd'))
        change_7d = safe_format(data['market_data']['price_change_percentage_7d_in_currency'].get('usd'))
        market_cap = safe_format(data['market_data']['market_cap'].get('usd'), "{:,.2f}")
        volume_24h = safe_format(data['market_data']['total_volume'].get('usd'), "{:,.2f}")
        circulating_supply = safe_format(data['market_data']['circulating_supply'], "{:,.0f}")
        max_supply = safe_format(data['market_data']['max_supply'], "{:,.0f}")
        fully_diluted_valuation = safe_format(data['market_data']['fully_diluted_valuation'].get('usd'), "{:,.2f}")  # Thêm FDV

        message = (
            f"📊 *Thông tin về {data['name']} ({data['symbol'].upper()})*:\n"
            f"💰 Giá hiện tại: *${price_usd}*\n"
            f"🔺 Giá cao nhất 24h: *${high_24h}*\n"
            f"🚀 Giá cao nhất mọi thời đại: *${all_time_high}*\n"
            f"📈 Thay đổi giá (1 giờ): *{change_1h}%*\n"
            f"📈 Thay đổi giá (24 giờ): *{change_24h}%*\n"
            f"📈 Thay đổi giá (7 ngày): *{change_7d}%*\n"
            f"🏦 Vốn hóa thị trường: *${market_cap}*\n"
            f"💎 Vốn hóa pha loãng hoàn toàn (FDV): *${fully_diluted_valuation}*\n"  # Hiển thị FDV
            f"📊 Khối lượng giao dịch 24 giờ: *${volume_24h}*\n"
            f"🔄 Lượng tiền đang lưu thông: *{circulating_supply} {data['symbol'].upper()}*\n"
            f"🛑 Nguồn cung tối đa: *{max_supply} {data['symbol'].upper()}*\n"
        )

        await update.message.reply_text(message, parse_mode="Markdown")

    except Exception as e:
        await update.message.reply_text(f"Đã xảy ra lỗi: {e}")


TIMEFRAME_MAPPING = {
    "1h": "price_change_percentage_1h_in_currency",
    "1d": "price_change_percentage_24h_in_currency",
    "1w": "price_change_percentage_7d_in_currency"
}

async def send_heatmap(chat, timeframe: str):
    """Tạo và gửi heatmap có màu giống hình mẫu"""
    try:
        print(f"📌 Đang tạo heatmap cho: {timeframe}")

        url = "https://api.coingecko.com/api/v3/coins/markets"
        params = {
            "vs_currency": "usd",
            "order": "market_cap_desc",
            "per_page": 100,
            "page": 1,
            "sparkline": False,
            "price_change_percentage": "1h,24h,7d"
        }
        response = requests.get(url, params=params)
        data = response.json()

        if response.status_code != 200 or not data:
            await chat.send_message("❌ Không thể lấy dữ liệu từ CoinGecko. Vui lòng thử lại sau!")
            return

        price_change_column = TIMEFRAME_MAPPING.get(timeframe)
        if price_change_column is None:
            await chat.send_message("⚠️ Sai khung thời gian! Vui lòng chọn 1h, 1d hoặc 1w.")
            return

        df = pd.DataFrame(data)
        if price_change_column not in df.columns:
            await chat.send_message(f"❌ API không trả về dữ liệu cho `{timeframe}`. Vui lòng thử lại sau!")
            return

        df["price_change"] = df[price_change_column]
        df = df.dropna(subset=["price_change"])

        # 🔹 Sắp xếp theo vốn hóa thị trường lớn nhất → nhỏ nhất
        df = df.sort_values("market_cap", ascending=False)

        # 🔹 Dùng sqrt(vốn hóa) để giảm chênh lệch kích thước
        df["size"] = np.sqrt(df["market_cap"])

        # 🔹 Dùng abs(price_change) để làm giá trị màu (càng lớn màu càng đậm)
        df["color_intensity"] = np.abs(df["price_change"])

        # 🔹 Chỉnh hệ màu giống như heatmap mẫu
        colorscale = [
            [0, "rgb(153, 0, 0)"],  # Đỏ đậm (giảm rất mạnh)
            [0.3, "rgb(204, 0, 0)"],  # Đỏ trung bình (giảm)
            [0.5, "rgb(255, 102, 102)"],  # Đỏ nhạt (giảm nhẹ)
            [0.5, "rgb(102, 255, 102)"],  # Xanh nhạt (tăng nhẹ)
            [0.7, "rgb(0, 204, 0)"],  # Xanh trung bình (tăng)
            [1, "rgb(0, 102, 0)"]   # Xanh đậm (tăng rất mạnh)
        ]

        # 🔹 Căn chỉnh văn bản đều trong từng ô
        df["text"] = df.apply(lambda row: f"<b>{row['symbol'].upper()}</b><br>${row['current_price']:,.2f}<br>{row['price_change']:.2f}%", axis=1)

        fig = go.Figure(data=go.Treemap(
            labels=df["symbol"].str.upper(),
            parents=[""] * len(df),
            values=df["size"],
            text=df["text"],
            textinfo="text",
            texttemplate="%{text}",
            marker=dict(
                colors=df["price_change"],
                colorscale=colorscale,
                cmid=0,
                showscale=True
            )
        ))

        fig.update_layout(
            title=f"📊 Heatmap top 100 coins ({timeframe.upper()})",
            template="plotly_dark"
        )

        html_path = f"heatmap_{timeframe}.html"
        fig.write_html(html_path)

        if not os.path.exists(html_path):
            await chat.send_message(f"❌ Lỗi khi tạo file heatmap_{timeframe}.html. Vui lòng thử lại!")
            return
        else:
            print(f"✅ File {html_path} đã được tạo thành công!")

        await chat.send_document(document=open(html_path, "rb"), filename=html_path)

        # Xóa file sau khi gửi xong (chờ 10 giây)
        await asyncio.sleep(10)
        os.remove(html_path)
        print(f"🗑️ File {html_path} đã được xóa.")

    except Exception as e:
        await chat.send_message(f"❌ Đã xảy ra lỗi: {e}")

async def heatmap(update, context):
    """Lệnh /heatmap tự động gửi 3 heatmap (1h, 1d, 1w) với màu sắc theo mẫu"""
    await update.message.reply_text("📊 Đang tạo heatmap 1h, 1d, 1w. Vui lòng chờ...")
    
    await send_heatmap(update.effective_chat, "1h")
    await send_heatmap(update.effective_chat, "1d")
    await send_heatmap(update.effective_chat, "1w")


async def desc(update, context):
    """Lấy thông tin chi tiết về đồng coin từ CoinGecko (bao gồm website và community)."""
    try:
        if not context.args:
            await update.message.reply_text("Vui lòng cung cấp mã coin. Ví dụ: /desc BTC")
            return

        coin_symbol = context.args[0].lower()

        # 🔹 Gọi API CoinGecko để lấy dữ liệu
        url_coingecko = f"https://api.coingecko.com/api/v3/coins/{coin_symbol}?localization=false"
        response_coingecko = requests.get(url_coingecko)

        if response_coingecko.status_code != 200:
            await update.message.reply_text(f"Không tìm thấy thông tin cho {coin_symbol}. Vui lòng kiểm tra lại.")
            return

        data_coingecko = response_coingecko.json()
        coin_name = data_coingecko.get("name", "Không có thông tin")
        symbol = data_coingecko.get("symbol", "N/A").upper()
        categories = ", ".join(data_coingecko.get("categories", ["Không có thông tin"]))

        # Lấy mô tả tiếng Việt nếu có, nếu không thì lấy mô tả tiếng Anh
        description_vi = data_coingecko["description"].get("vi")
        description_en = data_coingecko["description"].get("en")
        description = description_vi if description_vi else description_en if description_en else "Không có mô tả."

        # 🔹 Lấy thông tin website
        website = data_coingecko.get("links", {}).get("homepage", ["Không có thông tin"])[0]

        # 🔹 Lấy thông tin cộng đồng (hiển thị link)
        community_links = []
        links = data_coingecko.get("links", {})

        if links.get("twitter_screen_name"):
            community_links.append(f"❌ [X](https://twitter.com/{links['twitter_screen_name']})")
        if links.get("facebook_username"):
            community_links.append(f"Ⓕ [Facebook](https://www.facebook.com/{links['facebook_username']})")
        if links.get("telegram_channel_identifier"):
            community_links.append(f"📢 [Telegram](https://t.me/{links['telegram_channel_identifier']})")
        if links.get("subreddit_url"):
            community_links.append(f"Ⓡ [Reddit]({links['subreddit_url']})")
        if links.get("discord_url"):
            community_links.append(f"🎮 [Discord]({links['discord_url']})")

        community = "\n".join(community_links) if community_links else "Không có thông tin"

        # 🔹 Định dạng lại thông tin
        message = (
            f"*{coin_name} - ${symbol}*\n\n"
            f"📌 *Danh mục*: {categories}\n\n"
            f"📖 *Mô tả*: {description}\n\n"
            f"🌐 *Website*: {website}\n"
            f"🏛️ *Cộng đồng*:\n{community}"
        )

        await update.message.reply_text(message, parse_mode="Markdown", disable_web_page_preview=True)

    except Exception as e:
        await update.message.reply_text(f"Đã xảy ra lỗi: {e}")


async def set_webhook(application: Application):
    """Thiết lập Webhook."""
    await application.bot.set_webhook(WEBHOOK_URL)

def main():
    # Lấy cổng từ biến môi trường hoặc sử dụng cổng mặc định
    port = int(os.getenv("PORT", 8080))
    print(f"Đang sử dụng cổng: {port}")  # Log kiểm tra cổng

    # Khởi tạo ứng dụng Telegram bot
    application = Application.builder().token(TOKEN).build()

    # Đăng ký các handler
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("chart", chart))
    application.add_handler(CommandHandler("signal", signal))
    application.add_handler(CommandHandler("top", top))  # Thêm handler cho /top
    application.add_handler(CommandHandler("list", list_signals))
    application.add_handler(CommandHandler("smarttrade", current_price))  # Thêm handler cho /cap
    application.add_handler(CommandHandler("info", info))
    application.add_handler(CallbackQueryHandler(button))  # Thêm handler cho nút bấm từ /top
    application.add_handler(CommandHandler("heatmap", heatmap))
    application.add_handler(CommandHandler("desc", desc))

    # Chạy webhook
    application.run_webhook(
        listen="0.0.0.0",
        port=port,
        webhook_url=WEBHOOK_URL
    )

if __name__ == "__main__":
    main()





