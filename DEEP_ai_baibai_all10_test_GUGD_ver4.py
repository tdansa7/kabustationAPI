import requests
import json
import pandas as pd
from datetime import datetime, time, timedelta
import time as time_module
import os
from typing import Dict, List, Tuple
import logging

# ログ設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class KabuStationTrader:
    def __init__(self, api_password: str, api_url: str = "http://localhost:18080/kabusapi"):
        """
        kabuステーションAPIを使用した取引シミュレーター
        
        Args:
            api_password: APIパスワード
            api_url: APIのベースURL
        """
        self.api_url = api_url
        self.api_password = api_password
        self.token = None
        self.long_positions = {}  # 買いポジション
        self.short_positions = {}  # 空売りポジション
        self.trades = []  # 取引履歴
        self.watching_stocks = {}  # 監視中の銘柄
        self.signal_history = {}  # シグナル履歴
        self.profit_target = 0.04  # 利益確定ライン（+2%）
        self.stop_loss = -0.04  # 損切りライン（-1%）
        self.signal_window = 3  # シグナル有効期間（分）
        self.min_consecutive_signals = 2  # 必要な連続シグナル数
        self.limit_proximity_threshold = 0.05  # ストップ高・安まで5%の閾値
        self.ranking_update_interval_minutes = 5  # ランキング更新間隔（分）
        self.max_websocket_symbols = 50  # WebSocket登録可能最大銘柄数
        self.registered_symbols = set()  # 現在登録中の銘柄セット
        
    def get_token(self) -> str:
        """APIトークンを取得"""
        url = f"{self.api_url}/token"
        headers = {"Content-Type": "application/json"}
        data = {"APIPassword": self.api_password}
        
        try:
            response = requests.post(url, headers=headers, data=json.dumps(data))
            response.raise_for_status()
            self.token = response.json()["Token"]
            return self.token
        except Exception as e:
            logger.error(f"トークン取得エラー: {e}")
            raise
    
    def calculate_daily_limits(self, base_price: float) -> Tuple[float, float]:
        """
        基準価格から値幅制限（ストップ高・ストップ安）を計算
        
        Args:
            base_price: 基準価格（通常は前日終値）
            
        Returns:
            Tuple[float, float]: (ストップ高価格, ストップ安価格)
        """
        if base_price < 100:
            limit = 30
        elif base_price < 200:
            limit = 50
        elif base_price < 500:
            limit = 80
        elif base_price < 700:
            limit = 100
        elif base_price < 1000:
            limit = 150
        elif base_price < 1500:
            limit = 300
        elif base_price < 2000:
            limit = 400
        elif base_price < 3000:
            limit = 500
        elif base_price < 5000:
            limit = 700
        elif base_price < 7000:
            limit = 1000
        elif base_price < 10000:
            limit = 1500
        elif base_price < 15000:
            limit = 3000
        elif base_price < 20000:
            limit = 4000
        elif base_price < 30000:
            limit = 5000
        elif base_price < 50000:
            limit = 7000
        elif base_price < 70000:
            limit = 10000
        elif base_price < 100000:
            limit = 15000
        else:
            limit = 30000
        
        stop_high = base_price + limit
        stop_low = base_price - limit
        
        return stop_high, stop_low
    
    def is_near_daily_limit(self, current_price: float, prev_close: float) -> bool:
        """
        現在価格がストップ高・ストップ安まで5%以内かチェック
        
        Args:
            current_price: 現在価格
            prev_close: 前日終値
            
        Returns:
            bool: True if within 5% of daily limits
        """
        stop_high, stop_low = self.calculate_daily_limits(prev_close)
        
        # ストップ高までの距離
        distance_to_high = (stop_high - current_price) / (stop_high - prev_close) if stop_high != prev_close else 1
        # ストップ安までの距離
        distance_to_low = (current_price - stop_low) / (prev_close - stop_low) if prev_close != stop_low else 1
        
        # 5%以内の場合はTrue
        return distance_to_high <= self.limit_proximity_threshold or distance_to_low <= self.limit_proximity_threshold
    
    def get_rankings(self, ranking_type: str, target_count: int = 25) -> List[Dict]:
        """
        ランキング情報を取得（ストップ高・安に近い銘柄を除外）
        
        Args:
            ranking_type: "1" (値上がり率) or "2" (値下がり率)
            target_count: 取得したい銘柄数
        """
        url = f"{self.api_url}/ranking"
        headers = {"X-API-KEY": self.token}
        params = {
            "type": ranking_type,
            #"ExchangeDivision": "ALL"  # 全市場T
            "ExchangeDivision": "T" # 東証
        }
        
        try:
            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()
            all_rankings = response.json()["Ranking"]
            
            filtered_rankings = []
            processed_count = 0
            
            for stock in all_rankings:
                processed_count += 1
                symbol = stock["Symbol"]
                current_price = stock.get("CurrentPrice", 0)
                prev_close = stock.get("PrevClose", current_price)  # 前日終値がない場合は現在価格を使用
                
                # 価格が0の場合はスキップ
                if current_price <= 0 or prev_close <= 0:
                    logger.warning(f"{symbol}: 価格データが不正 (現在価格: {current_price}, 前日終値: {prev_close})")
                    continue
                
                # ストップ高・安に近いかチェック
                if self.is_near_daily_limit(current_price, prev_close):
                    stop_high, stop_low = self.calculate_daily_limits(prev_close)
                    logger.info(f"{symbol}: ストップ高・安に接近のため除外 (現在価格: {current_price}, ストップ高: {stop_high}, ストップ安: {stop_low})")
                    continue
                
                # フィルタを通過した銘柄を追加
                filtered_rankings.append(stock)
                
                # 目標数に達したら終了
                if len(filtered_rankings) >= target_count:
                    break
                
                # 処理した銘柄数が多すぎる場合は警告
                if processed_count >= target_count * 3:
                    logger.warning(f"多くの銘柄を処理しましたが、目標数 {target_count} に到達しませんでした。現在 {len(filtered_rankings)} 銘柄を取得。")
                    break
            
            logger.info(f"ランキング取得完了: {len(filtered_rankings)}/{processed_count} 銘柄 (タイプ: {ranking_type})")
            return filtered_rankings
            
        except Exception as e:
            logger.error(f"ランキング取得エラー: {e}")
            return []
    
    def get_board_info(self, symbol: str, exchange: str = "1") -> Dict:
        """板情報を取得"""
        url = f"{self.api_url}/board/{symbol}@{exchange}"
        headers = {"X-API-KEY": self.token}
        
        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"板情報取得エラー ({symbol}): {e}")
            return {}
    
    def unregister_websocket(self, symbols: List[str] = None) -> bool:
        """WebSocket配信銘柄を登録解除（全銘柄または指定銘柄）"""
        url = f"{self.api_url}/unregister"
        headers = {
            "Content-Type": "application/json",
            "X-API-KEY": self.token
        }
        
        if symbols is None:
            # 全銘柄を登録解除
            data = {"Symbols": []}
            self.registered_symbols.clear()
            logger.info("全銘柄のWebSocket登録を解除しました")
            url = f"{self.api_url}/unregister/all"
            try:
                response = requests.put(url, headers=headers)
                response.raise_for_status()
                return True
            except Exception as e:
                logger.error(f"WebSocket登録解除エラー: {e}")
                return False
        else:
            # 指定銘柄のみ登録解除
            symbols_data = [{"Symbol": s, "Exchange": 1} for s in symbols]
            data = {"Symbols": symbols_data}
            for symbol in symbols:
                self.registered_symbols.discard(symbol)
            logger.info(f"指定銘柄のWebSocket登録を解除しました: {symbols}")
        
        try:
            response = requests.put(url, headers=headers, data=json.dumps(data))
            response.raise_for_status()
            return True
        except Exception as e:
            logger.error(f"WebSocket登録解除エラー: {e}")
            return False
    
    def register_websocket(self, symbols: List[str]) -> bool:
        """WebSocket配信銘柄を登録"""
        url = f"{self.api_url}/register"
        headers = {
            "Content-Type": "application/json",
            "X-API-KEY": self.token
        }
        
        symbols_data = [{"Symbol": s, "Exchange": 1} for s in symbols]
        data = {"Symbols": symbols_data}
        
        try:
            response = requests.put(url, headers=headers, data=json.dumps(data))
            response.raise_for_status()
            # 登録成功した銘柄を記録
            for symbol in symbols:
                self.registered_symbols.add(symbol)
            logger.info(f"WebSocket登録成功: {len(symbols)}銘柄")
            return True
        except Exception as e:
            logger.error(f"WebSocket登録エラー: {e}")
            return False
    
    def get_position_symbols(self) -> List[str]:
        """現在のポジション銘柄を取得"""
        position_symbols = []
        
        # 買いポジション
        for symbol in self.long_positions.keys():
            position_symbols.append(symbol)
        
        # 空売りポジション
        for symbol in self.short_positions.keys():
            position_symbols.append(symbol)
        
        # 重複除去
        return list(set(position_symbols))
    
    def calculate_ranking_symbol_count(self, position_symbols: List[str]) -> Tuple[int, int]:
        """
        ランキングから取得する銘柄数を計算
        
        Returns:
            Tuple[int, int]: (急騰銘柄数, 急落銘柄数)
        """
        position_count = len(position_symbols)
        available_slots = self.max_websocket_symbols - position_count
        
        if available_slots <= 0:
            logger.warning(f"ポジション銘柄数({position_count})がWebSocket制限({self.max_websocket_symbols})を超えています")
            return 0, 0
        
        # 残り枠を半分ずつに分配
        rising_count = available_slots // 2
        falling_count = available_slots - rising_count
        
        logger.info(f"WebSocket銘柄配分: ポジション{position_count}, 急騰{rising_count}, 急落{falling_count}, 合計{position_count + rising_count + falling_count}")
        
        return rising_count, falling_count
    
    def update_websocket_registration(self, rising_stocks: List[Dict], falling_stocks: List[Dict]):
        """WebSocket登録を更新（既存ポジション優先）"""
        # 既存ポジション銘柄を取得
        position_symbols = self.get_position_symbols()
        
        # ランキングから取得する銘柄数を計算
        rising_count, falling_count = self.calculate_ranking_symbol_count(position_symbols)
        
        # ランキング銘柄を制限内で取得
        selected_rising = rising_stocks[:rising_count] if rising_count > 0 else []
        selected_falling = falling_stocks[:falling_count] if falling_count > 0 else []
        
        # 新しい登録対象銘柄リストを作成
        new_symbols = position_symbols.copy()
        for stock in selected_rising + selected_falling:
            if stock["Symbol"] not in new_symbols:
                new_symbols.append(stock["Symbol"])
        
        # WebSocket登録を更新
        if len(new_symbols) > self.max_websocket_symbols:
            logger.warning(f"登録対象銘柄数({len(new_symbols)})が制限({self.max_websocket_symbols})を超えました。制限内に調整します。")
            new_symbols = new_symbols[:self.max_websocket_symbols]
        
        # 現在の登録との差分を計算
        current_symbols = self.registered_symbols.copy()
        new_symbols_set = set(new_symbols)
        
        # 登録解除が必要な銘柄
        symbols_to_unregister = current_symbols - new_symbols_set
        # 新規登録が必要な銘柄
        symbols_to_register = new_symbols_set - current_symbols
        
        logger.info(f"WebSocket更新: 解除{len(symbols_to_unregister)}銘柄, 追加{len(symbols_to_register)}銘柄")
        
        # 差分がある場合のみ更新
        if symbols_to_unregister or symbols_to_register:
            # 全解除して再登録（差分管理よりシンプルで確実）
            self.unregister_websocket()
            
            # 新しい銘柄リストで登録
            if new_symbols:
                self.register_websocket(new_symbols)
        
        return selected_rising, selected_falling
    
    def display_websocket_status(self):
        """現在のWebSocket登録状況を表示"""
        position_symbols = self.get_position_symbols()
        registered_count = len(self.registered_symbols)
        position_count = len(position_symbols)
        
        print(f"\n=== WebSocket登録状況 ===")
        print(f"登録済み銘柄数: {registered_count}/{self.max_websocket_symbols}")
        print(f"ポジション銘柄数: {position_count}")
        print(f"ランキング銘柄数: {registered_count - position_count}")
        print(f"利用可能枠: {self.max_websocket_symbols - registered_count}")
        
        if position_symbols:
            print(f"ポジション銘柄: {', '.join(position_symbols)}")
        
        ranking_symbols = self.registered_symbols - set(position_symbols)
        if ranking_symbols:
            print(f"ランキング銘柄: {', '.join(list(ranking_symbols)[:10])}{'...' if len(ranking_symbols) > 10 else ''}")
    
    def check_volume_increase(self, symbol: str, current_data: Dict) -> bool:
        """1分間の出来高が上昇しているか確認"""
        current_volume = current_data.get("TradingVolume", 0)
        current_time = datetime.now()
        
        if symbol not in self.watching_stocks:
            self.watching_stocks[symbol] = {
                "volume": current_volume,
                "last_update": current_time,
                "volume_history": []  # 1分ごとの出来高履歴
            }
            return False
        
        # 前回更新からの経過時間
        time_diff = current_time - self.watching_stocks[symbol]["last_update"]
        
        if time_diff.seconds >= 60:
            # 1分間の出来高を計算（現在の累積出来高 - 1分前の累積出来高）
            prev_total_volume = self.watching_stocks[symbol]["volume"]
            volume_in_minute = current_volume - prev_total_volume
            
            # 履歴に追加
            self.watching_stocks[symbol]["volume_history"].append({
                "time": current_time,
                "volume": volume_in_minute
            })
            
            # 履歴を最新10分間に制限
            if len(self.watching_stocks[symbol]["volume_history"]) > 10:
                self.watching_stocks[symbol]["volume_history"].pop(0)
            
            # 前回の1分間出来高と比較
            if len(self.watching_stocks[symbol]["volume_history"]) >= 2:
                prev_minute_volume = self.watching_stocks[symbol]["volume_history"][-2]["volume"]
                current_minute_volume = volume_in_minute
                
                # 更新
                self.watching_stocks[symbol]["volume"] = current_volume
                self.watching_stocks[symbol]["last_update"] = current_time
                
                # 1分間の出来高が増加しているかチェック
                is_increasing = current_minute_volume > prev_minute_volume
                
                if is_increasing:
                    logger.info(f"{symbol}: 1分間出来高 {prev_minute_volume} → {current_minute_volume} (増加)")
                
                return is_increasing
            else:
                # 初回は比較対象がないのでFalse
                self.watching_stocks[symbol]["volume"] = current_volume
                self.watching_stocks[symbol]["last_update"] = current_time
                return False
        
        return False
    
    def check_high_update(self, symbol: str, current_data: Dict) -> bool:
        """1分前の高値を現在価格が更新しているか確認"""
        current_price = current_data.get("CurrentPrice", 0)
        current_high = current_data.get("HighPrice", 0)
        current_time = datetime.now()
        
        if symbol not in self.watching_stocks:
            self.watching_stocks[symbol] = {}
        
        if "high_history" not in self.watching_stocks[symbol]:
            self.watching_stocks[symbol]["high_history"] = []
            self.watching_stocks[symbol]["last_high_update"] = current_time
            self.watching_stocks[symbol]["last_recorded_high"] = current_high
            return False
        
        # 前回更新からの経過時間
        time_diff = current_time - self.watching_stocks[symbol]["last_high_update"]
        
        if time_diff.seconds >= 60:
            # 1分前の高値を記録
            prev_high = self.watching_stocks[symbol]["last_recorded_high"]
            
            # 履歴に追加
            self.watching_stocks[symbol]["high_history"].append({
                "time": self.watching_stocks[symbol]["last_high_update"],
                "high": prev_high,
                "current_price": current_price
            })
            
            # 履歴を最新10分間に制限
            if len(self.watching_stocks[symbol]["high_history"]) > 10:
                self.watching_stocks[symbol]["high_history"].pop(0)
            
            # 更新
            self.watching_stocks[symbol]["last_high_update"] = current_time
            self.watching_stocks[symbol]["last_recorded_high"] = current_high
            
            # 1分前の高値を現在価格が更新しているかチェック
            is_updated = current_price > prev_high
            
            if is_updated:
                logger.info(f"{symbol}: 高値更新 {prev_high} → {current_price} (1分前の高値を突破)")
            
            return is_updated
        
        return False
    
    def execute_trade(self, symbol: str, side: str, price: float, volume: int = 100, position_type: str = "long"):
        """取引を実行（シミュレーション）"""
        # 出来高情報を取得
        volume_info = None
        if symbol in self.watching_stocks and "volume_history" in self.watching_stocks[symbol]:
            history = self.watching_stocks[symbol]["volume_history"]
            if len(history) >= 2:
                volume_info = {
                    "prev_minute_volume": history[-2]["volume"],
                    "current_minute_volume": history[-1]["volume"],
                    "volume_change_rate": ((history[-1]["volume"] - history[-2]["volume"]) / 
                                         history[-2]["volume"] * 100) if history[-2]["volume"] > 0 else 0
                }
        
        # 高値情報を取得
        high_info = None
        if symbol in self.watching_stocks and "high_history" in self.watching_stocks[symbol]:
            high_history = self.watching_stocks[symbol]["high_history"]
            if high_history:
                latest = high_history[-1]
                high_info = {
                    "prev_high": latest["high"],
                    "break_price": latest["current_price"],
                    "break_rate": ((latest["current_price"] - latest["high"]) / 
                                 latest["high"] * 100) if latest["high"] > 0 else 0
                }
        
        # シグナル情報を取得
        signal_info = None
        if symbol in self.signal_history and len(self.signal_history[symbol]) >= 2:
            consecutive_count = len([s for s in self.signal_history[symbol] 
                                   if s["type"] in ["volume_high_combined", "volume_low_combined"]])
            signal_info = f"複合条件が{consecutive_count}回連続発生"
        
        trade = {
            "timestamp": datetime.now(),
            "symbol": symbol,
            "side": side,
            "position_type": position_type,
            "price": price,
            "volume": volume,
            "amount": price * volume,
            "volume_info": volume_info,
            "high_info": high_info,
            "signal_info": signal_info
        }
        
        if side == "buy":
            if position_type == "long":
                # 買いポジションを建てる
                if symbol not in self.long_positions:
                    self.long_positions[symbol] = {
                        "volume": 0,
                        "avg_price": 0,
                        "trades": []
                    }
                
                # ポジション更新
                total_volume = self.long_positions[symbol]["volume"] + volume
                total_amount = (self.long_positions[symbol]["avg_price"] * self.long_positions[symbol]["volume"] + 
                              price * volume)
                self.long_positions[symbol]["avg_price"] = total_amount / total_volume if total_volume > 0 else 0
                self.long_positions[symbol]["volume"] = total_volume
                self.long_positions[symbol]["trades"].append(trade)
                
            elif position_type == "short":
                # 空売りポジションを建てる
                if symbol not in self.short_positions:
                    self.short_positions[symbol] = {
                        "volume": 0,
                        "avg_price": 0,
                        "trades": []
                    }
                
                # ポジション更新
                total_volume = self.short_positions[symbol]["volume"] + volume
                total_amount = (self.short_positions[symbol]["avg_price"] * self.short_positions[symbol]["volume"] + 
                              price * volume)
                self.short_positions[symbol]["avg_price"] = total_amount / total_volume if total_volume > 0 else 0
                self.short_positions[symbol]["volume"] = total_volume
                self.short_positions[symbol]["trades"].append(trade)
                
        elif side == "sell":
            if position_type == "long" and symbol in self.long_positions:
                # 買いポジションを決済
                if self.long_positions[symbol]["volume"] >= volume:
                    # 利益計算（売値 - 買値）
                    profit = (price - self.long_positions[symbol]["avg_price"]) * volume
                    trade["profit"] = profit
                    trade["profit_rate"] = (price - self.long_positions[symbol]["avg_price"]) / self.long_positions[symbol]["avg_price"] * 100
                    
                    # ポジション更新
                    self.long_positions[symbol]["volume"] -= volume
                    if self.long_positions[symbol]["volume"] == 0:
                        del self.long_positions[symbol]
                        
            elif position_type == "short" and symbol in self.short_positions:
                # 空売りポジションを決済（買い戻し）
                if self.short_positions[symbol]["volume"] >= volume:
                    # 利益計算（売値 - 買値）※空売りなので逆
                    profit = (self.short_positions[symbol]["avg_price"] - price) * volume
                    trade["profit"] = profit
                    trade["profit_rate"] = (self.short_positions[symbol]["avg_price"] - price) / self.short_positions[symbol]["avg_price"] * 100
                    
                    # ポジション更新
                    self.short_positions[symbol]["volume"] -= volume
                    if self.short_positions[symbol]["volume"] == 0:
                        del self.short_positions[symbol]
        
        self.trades.append(trade)
        logger.info(f"取引実行: {symbol} {side} {position_type} {volume}株 @ {price}円")
    
    def check_low_update(self, symbol: str, current_data: Dict) -> bool:
        """1分前の安値を現在価格が更新しているか確認"""
        current_price = current_data.get("CurrentPrice", 0)
        current_low = current_data.get("LowPrice", 0)
        current_time = datetime.now()
        
        if symbol not in self.watching_stocks:
            self.watching_stocks[symbol] = {}
        
        if "low_history" not in self.watching_stocks[symbol]:
            self.watching_stocks[symbol]["low_history"] = []
            self.watching_stocks[symbol]["last_low_update"] = current_time
            self.watching_stocks[symbol]["last_recorded_low"] = current_low
            return False
        
        # 前回更新からの経過時間
        time_diff = current_time - self.watching_stocks[symbol]["last_low_update"]
        
        if time_diff.seconds >= 60:
            # 1分前の安値を記録
            prev_low = self.watching_stocks[symbol]["last_recorded_low"]
            
            # 履歴に追加
            self.watching_stocks[symbol]["low_history"].append({
                "time": self.watching_stocks[symbol]["last_low_update"],
                "low": prev_low,
                "current_price": current_price
            })
            
            # 履歴を最新10分間に制限
            if len(self.watching_stocks[symbol]["low_history"]) > 10:
                self.watching_stocks[symbol]["low_history"].pop(0)
            
            # 更新
            self.watching_stocks[symbol]["last_low_update"] = current_time
            self.watching_stocks[symbol]["last_recorded_low"] = current_low
            
            # 1分前の安値を現在価格が更新しているかチェック（下抜け）
            is_updated = current_price < prev_low
            
            if is_updated:
                logger.info(f"{symbol}: 安値更新 {prev_low} → {current_price} (1分前の安値を下抜け)")
            
            return is_updated
        
        return False
    
    def record_combined_signal(self, symbol: str, signal_type: str):
        """複合シグナル（両条件を同時に満たした状態）を履歴に記録"""
        current_time = datetime.now()
        
        if symbol not in self.signal_history:
            self.signal_history[symbol] = []
        
        # 複合シグナルを記録
        self.signal_history[symbol].append({
            "time": current_time,
            "type": signal_type
        })
        
        # 古いシグナルを削除（signal_window分以上前のもの）
        cutoff_time = current_time - timedelta(minutes=self.signal_window)
        self.signal_history[symbol] = [
            signal for signal in self.signal_history[symbol]
            if signal["time"] > cutoff_time
        ]
    
    def check_consecutive_combined_signals(self, symbol: str, signal_type: str, min_consecutive: int = 2) -> bool:
        """複合シグナルが連続して発生したかチェック"""
        if symbol not in self.signal_history:
            return False
        
        signals = self.signal_history[symbol]
        if len(signals) < min_consecutive:
            return False
        
        # 指定されたタイプのシグナルだけを抽出
        target_signals = [s for s in signals if s["type"] == signal_type]
        
        if len(target_signals) < min_consecutive:
            return False
        
        # 最新のシグナルから連続性をチェック
        consecutive_count = 1
        for i in range(len(target_signals) - 1, 0, -1):
            time_diff = target_signals[i]["time"] - target_signals[i-1]["time"]
            # 連続判定：前のシグナルから2分以内（1分間隔の監視で連続とみなす）
            if time_diff.total_seconds() <= 120:
                consecutive_count += 1
                if consecutive_count >= min_consecutive:
                    return True
            else:
                break
        
        return False
    
    def check_exit_conditions(self, symbol: str, current_price: float):
        """利益確定・損切り条件をチェック"""
        # 買いポジションのチェック
        if symbol in self.long_positions:
            avg_price = self.long_positions[symbol]["avg_price"]
            profit_rate = (current_price - avg_price) / avg_price
            
            if profit_rate >= self.profit_target:
                # 利益確定
                logger.info(f"{symbol}: 買いポジション利益確定 +{profit_rate*100:.1f}%")
                self.execute_trade(symbol, "sell", current_price, 
                                 self.long_positions[symbol]["volume"], "long")
            elif profit_rate <= self.stop_loss:
                # 損切り
                logger.info(f"{symbol}: 買いポジション損切り {profit_rate*100:.1f}%")
                self.execute_trade(symbol, "sell", current_price, 
                                 self.long_positions[symbol]["volume"], "long")
        
        # 空売りポジションのチェック
        if symbol in self.short_positions:
            avg_price = self.short_positions[symbol]["avg_price"]
            profit_rate = (avg_price - current_price) / avg_price
            
            if profit_rate >= self.profit_target:
                # 利益確定
                logger.info(f"{symbol}: 空売りポジション利益確定 +{profit_rate*100:.1f}%")
                self.execute_trade(symbol, "sell", current_price, 
                                 self.short_positions[symbol]["volume"], "short")
            elif profit_rate <= self.stop_loss:
                # 損切り
                logger.info(f"{symbol}: 空売りポジション損切り {profit_rate*100:.1f}%")
                self.execute_trade(symbol, "sell", current_price, 
                                 self.short_positions[symbol]["volume"], "short")
    
    def wait_for_market_open(self):
        """市場が開くまで待機"""
        while True:
            now = datetime.now()
            current_time = now.time()
            market_open = time(9, 0)
            market_close = time(15, 0)
            
            # 平日チェック（月曜日=0, 日曜日=6）
            if now.weekday() >= 5:
                # 週末の場合、月曜日まで待機
                days_until_monday = 7 - now.weekday()
                next_monday = now + timedelta(days=days_until_monday)
                next_market_open = datetime.combine(next_monday.date(), market_open)
                wait_seconds = (next_market_open - now).total_seconds()
                
                logger.info(f"週末です。月曜日の市場開始まで待機します。")
                logger.info(f"次の市場開始: {next_market_open.strftime('%Y-%m-%d %H:%M:%S')}")
                logger.info(f"待機時間: {wait_seconds/3600:.1f}時間")
                
                # 長時間待機の場合は1時間ごとに状態を確認
                while wait_seconds > 3600:
                    time_module.sleep(3600)
                    wait_seconds -= 3600
                    logger.info(f"市場開始まで残り: {wait_seconds/3600:.1f}時間")
                
                time_module.sleep(wait_seconds)
                continue
            
            # 平日の場合
            if current_time < market_open:
                # 当日の市場開始前
                market_open_datetime = datetime.combine(now.date(), market_open)
                wait_seconds = (market_open_datetime - now).total_seconds()
                
                logger.info(f"市場開始まで待機します。")
                logger.info(f"現在時刻: {now.strftime('%Y-%m-%d %H:%M:%S')}")
                logger.info(f"市場開始: {market_open_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
                logger.info(f"待機時間: {wait_seconds/60:.1f}分")
                
                # 5分ごとに状態を表示
                while wait_seconds > 300:
                    time_module.sleep(300)
                    wait_seconds -= 300
                    logger.info(f"市場開始まで残り: {wait_seconds/60:.1f}分")
                
                time_module.sleep(wait_seconds)
                logger.info("市場が開きました！取引を開始します。")
                break
                
            elif current_time > market_close:
                # 当日の市場終了後、翌営業日まで待機
                next_day = now + timedelta(days=1)
                # 金曜日の場合は月曜日まで待機
                if now.weekday() == 4:  # 金曜日
                    next_day = now + timedelta(days=3)
                
                next_market_open = datetime.combine(next_day.date(), market_open)
                wait_seconds = (next_market_open - now).total_seconds()
                
                logger.info(f"本日の市場は終了しました。次の市場開始まで待機します。")
                logger.info(f"次の市場開始: {next_market_open.strftime('%Y-%m-%d %H:%M:%S')}")
                logger.info(f"待機時間: {wait_seconds/3600:.1f}時間")
                
                # 長時間待機の場合は1時間ごとに状態を確認
                while wait_seconds > 3600:
                    time_module.sleep(3600)
                    wait_seconds -= 3600
                    logger.info(f"市場開始まで残り: {wait_seconds/3600:.1f}時間")
                
                time_module.sleep(wait_seconds)
                continue
            else:
                # 市場時間内
                logger.info("市場は開いています。取引を継続します。")
                break
    
    def monitor_and_trade(self):
        """銘柄を監視して取引を実行"""
        # 市場が開くまで待機
        self.wait_for_market_open()
        time_module.sleep(30)
        
        # 急騰・急落銘柄を取得（制限を考慮して動的に決定）
        position_symbols = self.get_position_symbols()
        rising_count, falling_count = self.calculate_ranking_symbol_count(position_symbols)
        
        rising_stocks = self.get_rankings("1", rising_count) if rising_count > 0 else []
        falling_stocks = self.get_rankings("2", falling_count) if falling_count > 0 else []
        
        # 選択された銘柄をCSVに保存
        if rising_stocks or falling_stocks:
            self.save_ranking_symbols(rising_stocks, falling_stocks)
        
        # WebSocket登録を更新（既存ポジション優先）
        selected_rising, selected_falling = self.update_websocket_registration(rising_stocks, falling_stocks)
        
        logger.info(f"監視対象銘柄: ポジション{len(position_symbols)}銘柄, 急騰{len(selected_rising)}銘柄, 急落{len(selected_falling)}銘柄")
        
        # WebSocket登録状況を表示
        self.display_websocket_status()
        
        # ランキング更新の管理
        last_ranking_update = datetime.now()
        ranking_update_interval = timedelta(minutes=self.ranking_update_interval_minutes)  # 設定可能な間隔
        
        # 監視ループ
        while True:
            # 市場が開いているかチェック
            if not self.is_market_open():
            #if False:
                logger.info("市場が閉じました。次の市場開始まで待機します。")
                self.wait_for_market_open()
                time_module.sleep(30)
                
                # 市場が再開したら銘柄リストを更新
                position_symbols = self.get_position_symbols()
                rising_count, falling_count = self.calculate_ranking_symbol_count(position_symbols)
                
                rising_stocks = self.get_rankings("1", rising_count) if rising_count > 0 else []
                falling_stocks = self.get_rankings("2", falling_count) if falling_count > 0 else []
                
                # 選択された銘柄をCSVに保存（市場再開時）
                if rising_stocks or falling_stocks:
                    self.save_ranking_symbols(rising_stocks, falling_stocks)
                
                # WebSocket登録を更新（既存ポジション優先）
                selected_rising, selected_falling = self.update_websocket_registration(rising_stocks, falling_stocks)
                
                logger.info(f"銘柄リスト更新（市場再開）: ポジション{len(position_symbols)}銘柄, 急騰{len(selected_rising)}銘柄, 急落{len(selected_falling)}銘柄")
                
                # WebSocket登録状況を表示
                self.display_websocket_status()
                last_ranking_update = datetime.now()  # 更新時刻をリセット
                continue
            
            # 定期的にランキングを更新
            current_time = datetime.now()
            if current_time - last_ranking_update >= ranking_update_interval:
                logger.info(f"{self.ranking_update_interval_minutes}分経過：ランキングを更新します...")
                
                # 新しいランキングを取得
                position_symbols = self.get_position_symbols()
                rising_count, falling_count = self.calculate_ranking_symbol_count(position_symbols)
                
                new_rising_stocks = self.get_rankings("1", rising_count) if rising_count > 0 else []
                new_falling_stocks = self.get_rankings("2", falling_count) if falling_count > 0 else []
                
                # 選択された銘柄をCSVに保存（定期更新）
                if new_rising_stocks or new_falling_stocks:
                    self.save_ranking_symbols(new_rising_stocks, new_falling_stocks)
                
                # WebSocket登録を更新（既存ポジション優先）
                selected_rising, selected_falling = self.update_websocket_registration(new_rising_stocks, new_falling_stocks)
                
                # 銘柄リストを更新
                rising_stocks = selected_rising
                falling_stocks = selected_falling
                
                logger.info(f"銘柄リスト更新（定期更新）: ポジション{len(position_symbols)}銘柄, 急騰{len(selected_rising)}銘柄, 急落{len(selected_falling)}銘柄")
                
                # WebSocket登録状況を表示
                self.display_websocket_status()
                last_ranking_update = current_time
                
                # 監視対象が変わった銘柄の古いシグナル履歴をクリア
                current_symbols = self.registered_symbols.copy()
                symbols_to_clear = set(self.signal_history.keys()) - current_symbols
                for symbol in symbols_to_clear:
                    if symbol in self.signal_history:
                        del self.signal_history[symbol]
                        logger.info(f"{symbol}: 監視対象外となったためシグナル履歴をクリア")
            # まず既存ポジションの利益確定・損切りチェック
            all_positions = list(self.long_positions.keys()) + list(self.short_positions.keys())
            for symbol in all_positions:
                time_module.sleep(0.11)
                board_info = self.get_board_info(symbol)
                if board_info:
                    current_price = board_info.get("CurrentPrice", 0)
                    if current_price > 0:
                        self.check_exit_conditions(symbol, current_price)
            
            # 急騰銘柄の監視（買いシグナル）
            for stock in rising_stocks:
                time_module.sleep(0.11)
                symbol = stock["Symbol"]
                board_info = self.get_board_info(symbol)
                
                if not board_info:
                    continue
                
                # 個別条件チェック
                volume_increased = self.check_volume_increase(symbol, board_info)
                high_updated = self.check_high_update(symbol, board_info)
                
                # 両条件を同時に満たした場合に複合シグナルを記録
                if volume_increased and high_updated:
                    self.record_combined_signal(symbol, "volume_high_combined")
                    logger.info(f"{symbol}: 複合シグナル発生（出来高増加＆高値更新）")
                    
                    # 複合シグナルが連続して発生したかチェック
                    if self.check_consecutive_combined_signals(symbol, "volume_high_combined", self.min_consecutive_signals):
                        current_price = board_info.get("CurrentPrice", 0)
                        if current_price > 0 and symbol not in self.long_positions:
                            logger.info(f"{symbol}: 買いシグナル検出（複合条件が{self.min_consecutive_signals}回連続発生）")
                            self.execute_trade(symbol, "buy", current_price, 100, "long")
                            # エントリー後はシグナル履歴をクリア
                            self.signal_history[symbol] = []
            
            # 急落銘柄の監視（空売りシグナル）
            for stock in falling_stocks:
                time_module.sleep(0.11)
                symbol = stock["Symbol"]
                board_info = self.get_board_info(symbol)
                
                if not board_info:
                    continue
                
                # 個別条件チェック
                volume_increased = self.check_volume_increase(symbol, board_info)
                low_updated = self.check_low_update(symbol, board_info)
                
                # 両条件を同時に満たした場合に複合シグナルを記録
                if volume_increased and low_updated:
                    self.record_combined_signal(symbol, "volume_low_combined")
                    logger.info(f"{symbol}: 複合シグナル発生（出来高増加＆安値更新）")
                    
                    # 複合シグナルが連続して発生したかチェック
                    if self.check_consecutive_combined_signals(symbol, "volume_low_combined", self.min_consecutive_signals):
                        current_price = board_info.get("CurrentPrice", 0)
                        if current_price > 0 and symbol not in self.short_positions:
                            logger.info(f"{symbol}: 空売りシグナル検出（複合条件が{self.min_consecutive_signals}回連続発生）")
                            self.execute_trade(symbol, "buy", current_price, 100, "short")
                            # エントリー後はシグナル履歴をクリア
                            self.signal_history[symbol] = []
            
            # 1分待機
            time_module.sleep(59)
            
            # 次回ランキング更新までの残り時間をログ出力（5分おきに）
            time_until_next_update = ranking_update_interval - (datetime.now() - last_ranking_update)
            minutes_remaining = time_until_next_update.total_seconds() / 60
            
            # 5分の倍数でログ出力（ただし更新間隔より短い場合のみ）
            if (int(minutes_remaining) % 5 == 0 and 
                int(minutes_remaining) > 0 and 
                int(minutes_remaining) < self.ranking_update_interval_minutes):
                logger.info(f"次回ランキング更新まで: {int(minutes_remaining)}分")
    
    def is_market_open(self) -> bool:
        """市場が開いているか確認"""
        now = datetime.now()
        market_open = time(9, 0)
        market_close = time(15, 0)
        
        # 平日チェック（月曜日=0, 日曜日=6）
        if now.weekday() >= 5:
            return False
        
        # 時間チェック
        current_time = now.time()
        return market_open <= current_time <= market_close
    
    def save_ranking_symbols(self, rising_stocks: List[Dict], falling_stocks: List[Dict]):
        """選択された銘柄をタイムスタンプ付きCSVに保存"""
        if not rising_stocks and not falling_stocks:
            logger.warning("保存する銘柄がありません")
            return None
            
        current_time = datetime.now()
        timestamp = current_time.strftime("%Y%m%d_%H%M%S")
        filename = f"stock_list_GU\\ranking_symbols_{timestamp}.csv"
        
        # データを準備
        ranking_data = []
        
        # 急騰銘柄
        for i, stock in enumerate(rising_stocks, 1):
            ranking_data.append({
                "rank": i,
                "symbol": stock["Symbol"],
                "name": stock.get("SymbolName", ""),
                "current_price": stock.get("CurrentPrice", 0),
                "change_rate": stock.get("ChangeRatio", 0),
                "trading_volume": stock.get("TradingVolume", 0),
                "category": "急騰",
                "ranking_type": "値上がり率",
                "timestamp": current_time.strftime("%Y-%m-%d %H:%M:%S")
            })
        
        # 急落銘柄
        for i, stock in enumerate(falling_stocks, 1):
            ranking_data.append({
                "rank": i + 25,  # 急騰銘柄の後に続く番号
                "symbol": stock["Symbol"],
                "name": stock.get("SymbolName", ""),
                "current_price": stock.get("CurrentPrice", 0),
                "change_rate": stock.get("ChangeRatio", 0),
                "trading_volume": stock.get("TradingVolume", 0),
                "category": "急落",
                "ranking_type": "値下がり率",
                "timestamp": current_time.strftime("%Y-%m-%d %H:%M:%S")
            })
        
        # DataFrameに変換
        df = pd.DataFrame(ranking_data)
        
        # ディレクトリが存在しない場合は作成
        dir_path = os.path.dirname(filename)
        if dir_path:  # パスが空でない場合のみ作成
            os.makedirs(dir_path, exist_ok=True)
        
        # CSVに保存
        df.to_csv(filename, index=False, encoding="utf-8-sig")
        logger.info(f"ランキング銘柄を {filename} に保存しました (合計{len(df)}銘柄)")
        
        # 簡易サマリー表示
        rising_count = len(rising_stocks)
        falling_count = len(falling_stocks)
        print(f"\n=== ランキング銘柄保存 ===")
        print(f"保存時刻: {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"急騰銘柄: {rising_count}銘柄")
        print(f"急落銘柄: {falling_count}銘柄")
        print(f"合計: {rising_count + falling_count}銘柄")
        print(f"ファイル: {filename}")
        
        # 銘柄コードのみのリストも表示
        all_symbols = [stock["Symbol"] for stock in rising_stocks + falling_stocks]
        print(f"銘柄コード: {', '.join(all_symbols)}")
        
        # 銘柄コードのみのシンプルなCSVも作成
        symbols_filename = self.save_symbols_only(rising_stocks, falling_stocks)
        
        return filename
    
    def save_symbols_only(self, rising_stocks: List[Dict], falling_stocks: List[Dict]):
        """銘柄コードのみのシンプルなCSVも作成"""
        current_time = datetime.now()
        timestamp = current_time.strftime("%Y%m%d_%H%M%S")
        filename = f"stock_list_GU\\symbols_only_{timestamp}.csv"
        
        # 銘柄コードのみのデータを準備
        symbols_data = []
        
        # 急騰銘柄
        for stock in rising_stocks:
            symbols_data.append({
                "symbol": stock["Symbol"],
                "category": "急騰"
            })
        
        # 急落銘柄
        for stock in falling_stocks:
            symbols_data.append({
                "symbol": stock["Symbol"],
                "category": "急落"
            })
        
        # DataFrameに変換
        df = pd.DataFrame(symbols_data)
        
        # ディレクトリが存在しない場合は作成
        dir_path = os.path.dirname(filename)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        
        # CSVに保存
        df.to_csv(filename, index=False, encoding="utf-8-sig")
        logger.info(f"銘柄コードリストを {filename} に保存しました")
        
        return filename
    
    def save_results(self, filename: str = "stock_list_GU\\trading_results.csv"):
        """取引結果をCSVに保存"""
        if not self.trades:
            logger.warning("取引履歴がありません")
            return
        
        # DataFrameに変換
        df = pd.DataFrame(self.trades)
        
        # 出来高情報を展開
        if "volume_info" in df.columns:
            df["prev_minute_volume"] = df["volume_info"].apply(
                lambda x: x["prev_minute_volume"] if x and "prev_minute_volume" in x else None)
            df["current_minute_volume"] = df["volume_info"].apply(
                lambda x: x["current_minute_volume"] if x and "current_minute_volume" in x else None)
            df["volume_change_rate"] = df["volume_info"].apply(
                lambda x: x["volume_change_rate"] if x and "volume_change_rate" in x else None)
            # 元の辞書列は削除
            df = df.drop(columns=["volume_info"])
        
        # 高値情報を展開
        if "high_info" in df.columns:
            df["prev_high"] = df["high_info"].apply(
                lambda x: x["prev_high"] if x and "prev_high" in x else None)
            df["break_price"] = df["high_info"].apply(
                lambda x: x["break_price"] if x and "break_price" in x else None)
            df["break_rate"] = df["high_info"].apply(
                lambda x: x["break_rate"] if x and "break_rate" in x else None)
            # 元の辞書列は削除
            df = df.drop(columns=["high_info"])
        
        # シグナル情報を追加
        if "signal_info" not in df.columns:
            df["signal_info"] = None
        
        # 損益計算
        df["profit"] = df.apply(lambda x: x.get("profit", 0), axis=1)
        df["profit_rate"] = df.apply(lambda x: x.get("profit_rate", 0), axis=1)
        
        # 累積損益
        df["cumulative_profit"] = df["profit"].cumsum()
        
        # CSVに保存
        df.to_csv(filename, index=False, encoding="utf-8-sig")
        logger.info(f"取引結果を {filename} に保存しました")
        
        # サマリー表示
        total_trades = len(df)
        total_profit = df["profit"].sum()
        win_trades = len(df[df["profit"] > 0])
        win_rate = (win_trades / total_trades * 100) if total_trades > 0 else 0
        
        # 買い/空売り別の統計
        long_trades = df[df["position_type"] == "long"]
        short_trades = df[df["position_type"] == "short"]
        
        print(f"\n=== 取引サマリー ===")
        print(f"総取引数: {total_trades}")
        print(f"  - 買い取引: {len(long_trades)}")
        print(f"  - 空売り取引: {len(short_trades)}")
        print(f"勝率: {win_rate:.2f}%")
        print(f"総損益: ¥{total_profit:,.0f}")
        print(f"  - 買い損益: ¥{long_trades['profit'].sum():,.0f}" if len(long_trades) > 0 else "  - 買い損益: ¥0")
        print(f"  - 空売り損益: ¥{short_trades['profit'].sum():,.0f}" if len(short_trades) > 0 else "  - 空売り損益: ¥0")
        print(f"平均損益: ¥{(total_profit/total_trades):,.0f}" if total_trades > 0 else "平均損益: N/A")

def main():
    # 設定
    API_PASSWORD = os.environ.get("KABU_STATION_API_PASSWORD", "API_PASSWORD")
    
    # トレーダー初期化
    trader = KabuStationTrader(API_PASSWORD)
    
    try:
        # トークン取得
        trader.get_token()
        logger.info("APIトークン取得成功")
        
        # 継続的に取引監視を実行
        while True:
            try:
                logger.info("取引監視を開始します...")
                trader.monitor_and_trade()
            except KeyboardInterrupt:
                logger.info("ユーザーによる中断を検出しました。")
                break
            except Exception as e:
                logger.error(f"エラーが発生しました: {e}")
                logger.info("30秒後に再試行します...")
                time_module.sleep(30)
                
                # トークンを再取得
                try:
                    trader.get_token()
                    logger.info("APIトークン再取得成功")
                except Exception as token_error:
                    logger.error(f"トークン再取得エラー: {token_error}")
                    
    except KeyboardInterrupt:
        logger.info("プログラムを終了します...")
    finally:
        # 結果保存
        trader.save_results()
        
        # 残ポジションがある場合は警告
        if trader.long_positions or trader.short_positions:
            print("\n=== 未決済ポジション ===")
            
            if trader.long_positions:
                print("\n【買いポジション】")
                for symbol, pos in trader.long_positions.items():
                    print(f"{symbol}: {pos['volume']}株 @ 平均¥{pos['avg_price']:,.0f}")
            
            if trader.short_positions:
                print("\n【空売りポジション】")
                for symbol, pos in trader.short_positions.items():
                    print(f"{symbol}: {pos['volume']}株 @ 平均¥{pos['avg_price']:,.0f}")

if __name__ == "__main__":
    main()