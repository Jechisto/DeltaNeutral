import streamlit as st
import pandas as pd
import sqlite3
import plotly.express as px
import plotly.graph_objects as go
import ccxt
import time
import numpy as np
import statsmodels.api as sm
from datetime import datetime, timedelta
from st_aggrid import AgGrid, GridOptionsBuilder
from io import StringIO


# Initialize database
@st.cache_resource
def init_db():
    conn = sqlite3.connect('trading.db')
    c = conn.cursor()
    
    # Create tables
    c.execute('''CREATE TABLE IF NOT EXISTS trades (
                 id INTEGER PRIMARY KEY AUTOINCREMENT,
                 long_sym TEXT NOT NULL,
                 short_sym TEXT NOT NULL,
                 long_size REAL NOT NULL,
                 short_size REAL NOT NULL,
                 open_datetime DATETIME NOT NULL,
                 close_datetime DATETIME,
                 is_closed BOOLEAN DEFAULT 0)''')
    
    c.execute('''CREATE TABLE IF NOT EXISTS equity_snapshots (
                 id INTEGER PRIMARY KEY AUTOINCREMENT,
                 trade_id INTEGER NOT NULL,
                 timestamp DATETIME NOT NULL,
                 long_price REAL NOT NULL,
                 short_price REAL NOT NULL,
                 long_value REAL NOT NULL,
                 short_value REAL NOT NULL,
                 equity_usd REAL NOT NULL,
                 equity_pct REAL NOT NULL,
                 FOREIGN KEY(trade_id) REFERENCES trades(id))''')
    
    conn.commit()
    return conn

# Database connection
def get_db():
    return sqlite3.connect('trading.db')

# Page configuration
st.set_page_config(
    page_title="Pair Trading Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Navigation
st.sidebar.title("📈 Pair Trading Dashboard")
PAGES = {
    "Analýza pozice": "position_analysis",
    "Přehled všech pozic": "positions_overview",
    "Update pozic": "update_positions",
    "Managovat pozice": "manage_positions",
    "Pair Trading Scanner": "pair_scanner",
    "O aplikaci": "about"
}
page = st.sidebar.radio("Navigace", list(PAGES.keys()))

# Helper functions
exchange = ccxt.binance()

def fetch_ohlcv(symbol, since):
    all_data = []
    now = int(time.time() * 1000)
    while since < now:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe='5m', since=since, limit=1000)
            if not ohlcv:
                break
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            all_data.append(df)
            since = int(df['timestamp'].iloc[-1].timestamp() * 1000) + 1
            if df['timestamp'].iloc[-1] >= pd.Timestamp.now():
                break
        except Exception as e:
            st.error(f"⚠️ Chyba při načítání {symbol}: {e}")
            break
    return pd.concat(all_data).set_index('timestamp') if all_data else None

def format_number(num, format_type='usd'):
    if pd.isna(num):
        return "N/A"
    if format_type == 'usd':
        return f"${num:,.2f}"
    elif format_type == 'pct':
        return f"{num:.2f}%"
    else:
        return f"{num:.2f}"

def get_color(value):
    return "green" if value > 0 else "red" if value < 0 else "gray"

# Data loading functions
@st.cache_data(ttl=300)
def load_trades():
    conn = get_db()
    trades_df = pd.read_sql("SELECT id, long_sym, short_sym FROM trades", conn)
    trades_df["pair"] = trades_df["long_sym"] + " vs " + trades_df["short_sym"]
    conn.close()
    return trades_df

@st.cache_data(ttl=300)
def load_equity_snapshots(trade_id, start_date=None, end_date=None):
    conn = get_db()
    query = f"SELECT * FROM equity_snapshots WHERE trade_id = {trade_id}"
    
    if start_date:
        query += f" AND timestamp >= '{start_date}'"
    if end_date:
        query += f" AND timestamp <= '{end_date}'"
    
    query += " ORDER BY timestamp"
    df = pd.read_sql(query, conn)
    conn.close()
    return df

@st.cache_data(ttl=300)
def load_all_positions():
    conn = get_db()
    query = """
    SELECT 
        t.id as trade_id,
        t.long_sym || ' vs ' || t.short_sym as pair,
        MIN(es.timestamp) as entry_date,
        MAX(es.timestamp) as latest_date,
        first_snap.equity_usd as initial_equity,
        latest_snap.equity_usd as current_equity,
        latest_snap.equity_usd - first_snap.equity_usd as profit_usd,
        latest_snap.equity_pct - first_snap.equity_pct as profit_pct,
        latest_snap.long_price as current_long_price,
        latest_snap.short_price as current_short_price,
        (latest_snap.long_price / first_snap.long_price - 1) * 100 as long_change_pct,
        (latest_snap.short_price / first_snap.short_price - 1) * 100 as short_change_pct
    FROM trades t
    JOIN equity_snapshots es ON t.id = es.trade_id
    JOIN equity_snapshots first_snap ON t.id = first_snap.trade_id 
        AND first_snap.timestamp = (SELECT MIN(timestamp) FROM equity_snapshots WHERE trade_id = t.id)
    JOIN equity_snapshots latest_snap ON t.id = latest_snap.trade_id 
        AND latest_snap.timestamp = (SELECT MAX(timestamp) FROM equity_snapshots WHERE trade_id = t.id)
    GROUP BY t.id
    ORDER BY profit_pct DESC
    """
    df = pd.read_sql(query, conn)
    conn.close()
    return df

@st.cache_data(ttl=300)
def get_open_trades():
    conn = get_db()
    query = "SELECT id, long_sym || ' vs ' || short_sym as pair, long_size, short_size FROM trades WHERE is_closed = 0"
    df = pd.read_sql(query, conn)
    conn.close()
    return df.set_index('pair')['id'].to_dict() if not df.empty else {}

def new_trade_form():
    with st.form("new_trade", clear_on_submit=True):
        col1, col2 = st.columns(2)
        long_sym = col1.text_input("Long Symbol", value="BTC").strip().upper()
        short_sym = col2.text_input("Short Symbol", value="ETH").strip().upper()
        
        col3, col4 = st.columns(2)
        long_size = col3.number_input("Long Size (USD)", 0.0, 1000000.0, 10000.0)
        short_size = col4.number_input("Short Size (USD)", 0.0, 1000000.0, 10000.0)
        
        col5, col6 = st.columns(2)
        entry_date = col5.date_input("Datum vstupu", value=datetime.now().date())
        entry_time = col6.time_input("Čas vstupu", value=datetime.now().time(), key="entry_time")
        
        pair_display = f"{long_sym} vs {short_sym}"
        st.markdown(f"**Auto-generated pair:** `{pair_display}`")
        
        if st.form_submit_button("Create Trade"):
            if not long_sym or not short_sym:
                st.error("Symboly nesmí být prázdné")
                return
                
            if long_size <= 0 or short_size <= 0:
                st.error("Velikost pozice musí být větší než 0")
                return
            
            # Vytvořte open_datetime AŽ PO odeslání formuláře
            open_datetime = datetime.combine(entry_date, entry_time)
            st.write(f"open_datetime: {open_datetime}")

            conn = get_db()
            c = conn.cursor()
            try:
                c.execute('''INSERT INTO trades 
                    (long_sym, short_sym, long_size, short_size, open_datetime, is_closed) 
                    VALUES (?, ?, ?, ?, ?, ?)''', 
                    (long_sym, short_sym, long_size, short_size, open_datetime, False))
                trade_id = c.lastrowid
                conn.commit()
                st.success(f"Created new trade #{trade_id}")
            except Exception as e:
                st.error(f"Failed to create trade: {e}")
            finally:
                conn.close()

def close_position_form():
    open_trades = get_open_trades()
    
    if not open_trades:
        st.info("No open trades found")
        return
        
    selected_pair = st.selectbox("Select Trade to Close", options=list(open_trades.keys()))
    trade_id = open_trades[selected_pair]

    close_date = st.date_input("Datum ukončení", value=datetime.now().date())
    close_time = st.time_input("Čas ukončení", value=datetime.now().time())
    close_datetime = datetime.combine(close_date, close_time)

    if st.button("Close Selected Trade"):
        conn = get_db()
        try:
            conn.execute("UPDATE trades SET is_closed = 1, close_datetime = ? WHERE id = ?", 
                       (close_datetime, trade_id))
            conn.commit()
            st.success(f"Closed trade: {selected_pair}")
            # Invalidate cache
            st.cache_data.clear()
        except Exception as e:
            st.error(f"Failed to close trade: {e}")
        finally:
            conn.close()

# Page: Position Analysis
if page == "Analýza pozice":
    st.title("📈 Vývoj pozic podle obchodu")
    trades_df = load_trades()
    
    if trades_df.empty:
        st.warning("Žádné obchody nebyly nalezeny.")
        st.stop()
        
    selected_pair = st.selectbox("Vyber obchod (long vs short):", trades_df["pair"].unique())
    trade_id = trades_df[trades_df["pair"] == selected_pair]["id"].values[0]
    
    # Date range filter
    df = load_equity_snapshots(trade_id)
    min_timestamp = df["timestamp"].min() if not df.empty else None
    max_timestamp = df["timestamp"].max() if not df.empty else None
    min_timestamp = pd.to_datetime(min_timestamp) if pd.notnull(min_timestamp) else None
    max_timestamp = pd.to_datetime(max_timestamp) if pd.notnull(max_timestamp) else None
    min_date = min_timestamp.date() if min_timestamp is not None else datetime.now().date()
    max_date = max_timestamp.date() if max_timestamp is not None else datetime.now().date()
    
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Od data:", min_date)
        # Získání výchozího času z databáze (open_datetime)
        conn = get_db()
        trade_info = pd.read_sql(f"SELECT open_datetime FROM trades WHERE id = {trade_id}", conn)
        conn.close()
        if not trade_info.empty:
            default_start_time = pd.to_datetime(trade_info['open_datetime'].iloc[0]).time()
        else:
            default_start_time = min_timestamp.time() if min_timestamp is not None else datetime.now().time()
        start_time = st.time_input("Od času:", default_start_time, key="start_time")
        start_datetime = datetime.combine(start_date, start_time)
    with col2:
        end_date = st.date_input("Do data:", max_date)
        end_time = st.time_input("Do času:", max_timestamp.time() if max_timestamp is not None else datetime.now().time(), key="end_time")
        end_datetime = datetime.combine(end_date, end_time)

    # Use start_datetime and end_datetime for filtering
    start_date = start_datetime
    end_date = end_datetime
    
    if start_date > end_date:
        st.error("Datum 'Od' nemůže být větší než datum 'Do'")
        st.stop()
    
    df = load_equity_snapshots(trade_id, start_date, end_date)
    
    if df.empty:
        st.warning("Žádná data k zobrazení.")
        st.stop()
        
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    long_sym, short_sym = selected_pair.split(" vs ")[:2]
    
    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Equity", "💰 Ceny", "📋 Detaily", "📊 Z-score"])
    
    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            fig = px.line(df, x="timestamp", y="equity_usd", title=f"Equity USD - {selected_pair}")
            fig.update_layout(height=400, xaxis_title="", yaxis_title="Equity (USD)")
            st.plotly_chart(fig, use_container_width=True,key="equity_usd")
        
        with col2:
            fig2 = px.line(df, x="timestamp", y="equity_pct", title=f"Equity % - {selected_pair}")
            fig2.update_layout(height=400, xaxis_title="", yaxis_title="Equity (%)")
            st.plotly_chart(fig2, use_container_width=True, key="equity_pct")
    
    with tab2:
        # Normalized prices
        df['long_norm'] = df['long_price'] / df['long_price'].iloc[0]
        df['short_norm'] = df['short_price'] / df['short_price'].iloc[0]
        
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=df["timestamp"], y=df["long_norm"]))
        fig3.add_trace(go.Scatter(x=df["timestamp"], y=df["short_norm"]))
        fig3.update_layout(title=f"Normalizované ceny {selected_pair}", height=500)
        st.plotly_chart(fig3, use_container_width=True, key="long_short_norm")
        
        # # Price ratio
        # df['price_ratio'] = df['long_price'] / df['short_price']
        # fig_ratio = px.line(df, x="timestamp", y="price_ratio", title="Poměr cen (Long/Short)")
        # fig_ratio.update_layout(height=400)
        # st.plotly_chart(fig_ratio, use_container_width=True, key="price_ratio")

        # Z-score calculation
        df['z_score'] = (df['equity_pct'] - df['equity_pct'].mean()) / df['equity_pct'].std()
        
        fig6 = go.Figure()
        fig6.add_trace(go.Scatter(x=df["timestamp"], y=df["z_score"], name="Z-score"))
        fig6.add_hline(y=2, line_dash="dash", line_color="green", annotation_text="2.0")
        fig6.add_hline(y=0, line_dash="dash", line_color="blue", annotation_text="0.0")
        fig6.add_hline(y=-2, line_dash="dash", line_color="red", annotation_text="-2.0")
        fig6.update_layout(title=f"Z-score - {selected_pair}", height=500)
        st.plotly_chart(fig6, use_container_width=True, key="zscore_tab2")
        
        st.info("Z-score ukazuje, jak daleko je aktuální výkon od průměru v jednotkách směrodatné odchylky. "
                "Hodnoty nad 2 nebo pod -2 mohou signalizovat obchodní příležitosti.")
        
    
    with tab3:
        # Performance metrics
        first = df.iloc[0]
        last = df.iloc[-1]

        # Tabulka s hodnotami na začátku a na konci období
        st.subheader("Porovnání hodnot na začátku a na konci období")
        compare_df = pd.DataFrame({
            "Začátek": {
            "Equity (USD)": format_number(first["equity_usd"]),
            "Equity (%)": format_number(first["equity_pct"], 'pct'),
            f"Cena {long_sym}": format_number(first["long_price"]),
            f"Cena {short_sym}": format_number(first["short_price"]),
            },
            "Konec": {
            "Equity (USD)": format_number(last["equity_usd"]),
            "Equity (%)": format_number(last["equity_pct"], 'pct'),
            f"Cena {long_sym}": format_number(last["long_price"]),
            f"Cena {short_sym}": format_number(last["short_price"]),
            }
        })
        st.table(compare_df)
        
        # Výpočet procentuálního zisku/ztráty pro long a short normalizované ceny
        long_pct = (last['long_norm'] - 1) * 100  # zisk pokud > 1
        short_pct = (1 - last['short_norm']) * 100  # zisk pokud < 1
        total_pct = long_pct + short_pct

        st.info(f"Součet zisků/ztrát: Long: {long_pct:.2f} % + Short: {short_pct:.2f} % = **{total_pct:.2f} %**")

        usd_diff = last["equity_usd"] - first["equity_usd"]
        pct_diff = last["equity_pct"] - first["equity_pct"]
        long_change = (last["long_price"] / first["long_price"] - 1) * 100
        short_change = (last["short_price"] / first["short_price"] - 1) * 100

        profit = first["long_value"] * total_pct/100
        profit = round(profit, 2)

        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Profit (USD)", format_number(profit), delta=format_number(profit))
        col2.metric("Zisk/Ztráta (%)", f"{total_pct:.2f} %", delta=format_number(pct_diff, 'pct'))
        col3.metric(f"Změna {long_sym}", format_number(long_change, 'pct'), 
                   delta=format_number(long_change, 'pct'))
        col4.metric(f"Změna {short_sym}", format_number(short_change, 'pct'), 
                   delta=format_number(short_change, 'pct'))
        
        # Latest data
        st.subheader("Poslední data")
        st.dataframe(df.tail(10).set_index('timestamp'), use_container_width=True)
    
    with tab4:
        # Z-score calculation
        df['z_score'] = (df['equity_pct'] - df['equity_pct'].mean()) / df['equity_pct'].std()
        
        fig5 = go.Figure()
        fig5.add_trace(go.Scatter(x=df["timestamp"], y=df["z_score"], name="Z-score"))
        fig5.add_hline(y=2, line_dash="dash", line_color="green", annotation_text="2.0")
        fig5.add_hline(y=0, line_dash="dash", line_color="blue", annotation_text="0.0")
        fig5.add_hline(y=-2, line_dash="dash", line_color="red", annotation_text="-2.0")
        fig5.update_layout(title=f"Z-score - {selected_pair}", height=500)
        st.plotly_chart(fig5, use_container_width=True, key="zscore_tab4")
        
        st.info("Z-score ukazuje, jak daleko je aktuální výkon od průměru v jednotkách směrodatné odchylky. "
                "Hodnoty nad 2 nebo pod -2 mohou signalizovat obchodní příležitosti.")

# Page: Positions Overview
elif page == "Přehled všech pozic":
    st.title("📋 Přehled všech obchodních pozic")
    all_positions = load_all_positions()
    
    if all_positions.empty:
        st.warning("Žádné pozice k zobrazení.")
        st.stop()
    
    # Summary stats
    total_profit = all_positions["profit_usd"].iloc[-1] #.sum
    avg_profit = all_positions["profit_pct"].iloc[-1] #.mean
    profitable = sum(all_positions["profit_pct"] > 0)
    losing = sum(all_positions["profit_pct"] <= 0)
    win_rate = profitable / len(all_positions) * 100
    
    st.subheader("Souhrnné statistiky")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Celkový zisk", format_number(total_profit))
    col2.metric("Průměrný výnos", format_number(avg_profit, 'pct'))
    col3.metric("Ziskové obchody", f"{profitable}/{len(all_positions)}", delta=f"{win_rate:.1f}%")
    col4.metric("Ztrátové obchody", f"{losing}/{len(all_positions)}")
    
    # Performance chart
    st.subheader("Výkonnost obchodů")
    fig = px.bar(all_positions, x='pair', y='profit_pct', 
                 color='profit_pct', color_continuous_scale='RdYlGn',
                 title='Zisk/Ztráta podle páru (%)')
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)
    
    # Positions table
    st.subheader("Detaily pozic")
    gb = GridOptionsBuilder.from_dataframe(all_positions)
    gb.configure_pagination(paginationPageSize=15)
    gb.configure_side_bar()
    gridOptions = gb.build()
    AgGrid(all_positions, gridOptions=gridOptions, height=400, fit_columns_on_grid_load=True)

# Page: Update Positions
elif page == "Update pozic":
    st.title("🔄 Update pozic")
    st.info("Tato funkce stahuje aktuální data z Binance a aktualizuje vaše pozice.")
    
    if st.button("Aktualizovat všechny otevřené pozice"):
        open_trades = get_open_trades()
        
        if not open_trades:
            st.warning("Žádné otevřené pozice k aktualizaci.")
            st.stop()
            
        progress_bar = st.progress(0)
        status_text = st.empty()
        conn = get_db()
        
        for i, (pair, trade_id) in enumerate(open_trades.items()):
            status_text.text(f"Zpracovávám: {pair} ({i+1}/{len(open_trades)})")
            
            # Get symbols
            pair_str = str(pair)
            split_pair = pair_str.split(" vs ")
            if len(split_pair) != 2:
                st.error(f"Chybný formát páru: {pair_str}")
                continue
            long_sym, short_sym = split_pair
            long_symbol = f"{long_sym}/USDT"
            short_symbol = f"{short_sym}/USDT"
            # Získání velikostí pozic z databáze
            trade_info = pd.read_sql(
                f"SELECT long_size, short_size,open_datetime FROM trades WHERE id = {trade_id}",
                conn
            )
            if trade_info.empty:
                st.error(f"Nenalezeny informace o velikosti pozice pro {pair}")
                continue
            long_val = trade_info['long_size'].iloc[0]
            short_val = trade_info['short_size'].iloc[0]
            open = trade_info['open_datetime'].iloc[0]

            # Get last snapshot
            last_snapshot = pd.read_sql(
                f"SELECT MAX(timestamp) as last FROM equity_snapshots WHERE trade_id = {trade_id}", 
                conn
            )
            last_time = last_snapshot['last'].iloc[0]
            
            # Determine since parameter
            if last_time:
                since = int(pd.Timestamp(last_time).timestamp() * 1000) + 1
            else:
                start_date = datetime.now() - timedelta(days=7)
                since = int(start_date.timestamp() * 1000)
            
            # Fetch data
            long_data = fetch_ohlcv(long_symbol, since)
            short_data = fetch_ohlcv(short_symbol, since)
            
            if long_data is None or short_data is None:
                st.error(f"Chyba při načítání dat pro {pair}")
                continue
                
            # Merge data
            merged = pd.merge(
                long_data[['close']], 
                short_data[['close']], 
                left_index=True, 
                right_index=True,
                suffixes=('_long', '_short')
            )
            
            if merged.empty:
                st.warning(f"Žádná nová data pro {pair}")
                continue
                
            # Insert into DB
            for ts, row in merged.iterrows():
                timestamp = ts.strftime('%Y-%m-%d %H:%M:%S')
                long_price = row['close_long']
                short_price = row['close_short']
                
                
                # Simple equity calculation
                long_value = float(long_price) * float(long_val)
                short_value = float(short_price) * float(short_val)
                # Výpočet long_value ve stejném stylu jako: df['long_value'] = df['long'] / df['long'].iloc[0] * long_usd
                # Zde: long_value = (aktuální cena long / vstupní cena long) * velikost long pozice v USD
                # Vstupní cena long je třeba získat z databáze (první snapshot nebo open price)
                # Pokud není snapshot, použijeme cenu z prvního řádku merged
                # Získání vstupní ceny long
                entry_long_price = None
                entry_short_price = None

                entry_snapshot = pd.read_sql(
                    f"SELECT long_price, short_price,long_value,short_value FROM equity_snapshots WHERE trade_id = {trade_id} ORDER BY timestamp ASC LIMIT 1",
                    conn
                )
                if not entry_snapshot.empty:
                    entry_long_price = entry_snapshot['long_price'].iloc[0]
                    entry_short_price = entry_snapshot['short_price'].iloc[0]
                    entry_long_value = entry_snapshot['long_value'].iloc[0]
                    entry_short_value = entry_snapshot['short_value'].iloc[0]
                else:
                    entry_long_price = merged['close_long'].iloc[0]
                    entry_short_price = merged['close_short'].iloc[0]
                    entry_long_value = long_value
                    entry_short_value = short_value

                long_value = (long_price / entry_long_price) * long_val if entry_long_price else long_value
                # Short value: inverzní výpočet vůči long (když cena klesá, short vydělává)
                short_value = (entry_short_price / short_price) * short_val if entry_short_price else short_value

                equity_usd = (long_value + short_value) - (entry_long_value + entry_short_value )

                # Výpočet procentuálního zisku od vstupu
                # equity_pct = ((aktuální equity_usd / počáteční equity_usd) - 1) * 100
                # Získání počáteční equity_usd
                entry_equity = None
                entry_equity_row = pd.read_sql(
                    f"SELECT equity_usd FROM equity_snapshots WHERE trade_id = {trade_id} ORDER BY timestamp ASC LIMIT 1",
                    conn
                )
                if not entry_equity_row.empty:
                    entry_equity = entry_equity_row['equity_usd'].iloc[0]
                else:
                    entry_equity = long_value - short_value  # Pokud není, použij aktuální jako základ

                # Výpočet procentuální změny zvlášť pro long a short
                long_pct = ((long_value / entry_long_value) - 1) * 100 if entry_long_value else 0.0
                short_pct = ((short_value / entry_short_value) - 1) * 100 if entry_short_value else 0.0
                # Celková procentuální změna jako průměr obou
                equity_pct = (long_pct + short_pct) / 2
                
                conn.execute('''INSERT OR IGNORE INTO equity_snapshots 
                              (trade_id, timestamp, long_price, short_price, 
                               long_value, short_value, equity_usd, equity_pct)
                              VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
                              (trade_id, timestamp, long_price, short_price, 
                               long_value, short_value, equity_usd, equity_pct))
            
            conn.commit()
            progress_bar.progress((i+1)/len(open_trades))
        
        conn.close()
        st.success("Aktualizace dokončena!")

# Page: Manage Positions
elif page == "Managovat pozice":
    st.title("📈 Otevřít/zavřít obchodní pozici")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("➕ Nová pozice")
        new_trade_form()
    
    with col2:
        st.subheader("🔒 Uzavřít pozici")
        close_position_form()
    
    # Open positions table
    st.subheader("Aktivní pozice")
    open_trades = get_open_trades()
    if open_trades:
        # Načti detailní info o otevřených pozicích
        conn = get_db()
        df = pd.read_sql(
            f"SELECT id, long_sym || ' vs ' || short_sym as pair, open_datetime FROM trades WHERE is_closed = 0",
            conn
        )
        conn.close()
        if not df.empty:
            df['open_datetime'] = pd.to_datetime(df['open_datetime'])
            df['Datum vstupu'] = df['open_datetime'].dt.date
            df['Čas vstupu'] = df['open_datetime'].dt.time
            st.table(df[['pair', 'Datum vstupu', 'Čas vstupu']].rename(columns={'pair': 'Obchodní pár'}))
        else:
            st.info("Žádné aktivní pozice")
    else:
        st.info("Žádné aktivní pozice")

# Page: Pair Trading Scanner
elif page == "Pair Trading Scanner":
    st.title("🔍 Pair Trading Scanner")
    st.info("Tento nástroj hledá potenciální pair trading příležitosti na základě historických dat.")
    
    # Symbol selection
    st.subheader("1. Výběr symbolů")
    default_symbols = "BTC\nETH\nBNB\nSOL\nXRP\nADA\nDOT\nDOGE\nAVAX\nMATIC"
    symbols_text = st.text_area("Zadejte symboly (jeden na řádek):", value=default_symbols, height=150)
    symbols = [s.strip().upper() for s in symbols_text.split('\n') if s.strip()]
    
    if not symbols:
        st.warning("Zadejte alespoň 2 symboly")
        st.stop()
    
    # Date range
    st.subheader("2. Časové období")
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Počáteční datum", value=datetime.now() - timedelta(days=365))
    with col2:
        end_date = st.date_input("Koncové datum", value=datetime.now())
    
    # Parameters
    st.subheader("3. Parametry vyhledávání")
    min_correlation = st.slider("Minimální korelace", 0.7, 1.0, 0.9)
    max_half_life = st.slider("Maximální half-life (dny)", 1, 100, 30)
    
    if st.button("Hledat příležitosti"):
        if len(symbols) < 2:
            st.error("Potřebujete alespoň 2 symboly pro analýzu")
            st.stop()
            
        progress_bar = st.progress(0)
        status_text = st.empty()
        results = []
        
        # Fetch price data
        prices = {}
        for i, symbol in enumerate(symbols):
            status_text.text(f"Načítám data pro {symbol} ({i+1}/{len(symbols)})")
            symbol_pair = f"{symbol}/USDT"
            # Oprava: přidána chybějící závorka
            since = int(pd.Timestamp(start_date).timestamp()) * 1000
            data = fetch_ohlcv(symbol_pair, since)
            if data is not None:
                prices[symbol] = data['close']
            progress_bar.progress((i+1)/len(symbols))
        
        if len(prices) < 2:
            st.error("Nedostatek dat pro analýzu")
            st.stop()
            
        # Analyze pairs
        status_text.text("Analyzuji páry...")
        pairs = []
        for i in range(len(symbols)):
            for j in range(i+1, len(symbols)):
                sym1, sym2 = symbols[i], symbols[j]
                if sym1 in prices and sym2 in prices:
                    pairs.append((sym1, sym2))
        
        opportunities = []
        for i, (sym1, sym2) in enumerate(pairs):
            status_text.text(f"Analyzuji pár: {sym1} vs {sym2} ({i+1}/{len(pairs)})")
            
            # Merge prices
            df = pd.DataFrame({sym1: prices[sym1], sym2: prices[sym2]})
            df = df.dropna()
            
            if len(df) < 30:
                continue
                
            # Calculate spread
            spread = df[sym1] / df[sym2]
            
            # Calculate correlation
            correlation = df[sym1].corr(df[sym2])
            if correlation < min_correlation:
                continue
                
            # Calculate cointegration
            model = sm.OLS(df[sym1], sm.add_constant(df[sym2])).fit()
            hedge_ratio = model.params[sym2]
            spread = df[sym1] - hedge_ratio * df[sym2]
            
            # Calculate z-score
            mean = spread.mean()
            std = spread.std()
            z_score = (spread.iloc[-1] - mean) / std
            
            # Calculate half-life
            spread_lag = spread.shift(1)
            delta = spread - spread_lag
            delta = delta.dropna()
            spread_lag = spread_lag.dropna()
            model = sm.OLS(delta, sm.add_constant(spread_lag)).fit()
            lambda_ = model.params.iloc[1]
            half_life = -np.log(2) / lambda_ if lambda_ < 0 else float('inf')
            
            if half_life > max_half_life:
                continue
                
            opportunities.append({
                'pair': f"{sym1} vs {sym2}",
                'correlation': correlation,
                'z_score': z_score,
                'half_life': half_life,
                'current_spread': spread.iloc[-1],
                'mean_spread': mean,
                'std_spread': std
            })
            
            progress_bar.progress((i+1)/len(pairs))
        
        # Display results
        if opportunities:
            opportunities_df = pd.DataFrame(opportunities)
            st.subheader("Nalezené příležitosti")
            
            # Sort by z-score absolute value
            opportunities_df['abs_z'] = opportunities_df['z_score'].abs()
            opportunities_df = opportunities_df.sort_values('abs_z', ascending=False)
            
            # Format columns
            opportunities_df['correlation'] = opportunities_df['correlation'].apply(lambda x: f"{x:.4f}")
            opportunities_df['z_score'] = opportunities_df['z_score'].apply(lambda x: f"{x:.2f}")
            opportunities_df['half_life'] = opportunities_df['half_life'].apply(lambda x: f"{x:.1f} dní")
            
            st.dataframe(opportunities_df[['pair', 'correlation', 'z_score', 'half_life']], 
                         height=min(400, 50 * len(opportunities_df)))
            
            # Show details for selected pair
            selected_pair = st.selectbox("Vyberte pár pro detail", opportunities_df['pair'])
            selected = opportunities_df[opportunities_df['pair'] == selected_pair].iloc[0]
            
            st.subheader(f"Detail: {selected_pair}")
            col1, col2, col3 = st.columns(3)
            col1.metric("Korelace", selected['correlation'])
            col2.metric("Z-score", selected['z_score'])
            col3.metric("Half-life", selected['half_life'])
            
            # Show spread chart
            sym1, sym2 = selected_pair.split(" vs ")
            spread = prices[sym1] / prices[sym2]
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=spread.index, y=spread, name="Poměr cen"))
            fig.add_hline(y=spread.mean(), line_dash="dash", line_color="blue", annotation_text="Průměr")
            fig.update_layout(title=f"Poměr cen {selected_pair}", height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            # Trading signal
            z_score = float(selected['z_score'])
            if z_score > 2:
                st.success(f"Příležitost: Prodat {sym1} / Koupit {sym2} (Z-score = {z_score:.2f})")
            elif z_score < -2:
                st.success(f"Příležitost: Koupit {sym1} / Prodat {sym2} (Z-score = {z_score:.2f})")
            else:
                st.info("Žádný silný signál (|Z-score| < 2)")
        else:
            st.warning("Nebyla nalezena žádná vhodná příležitost.")

# Page: About
elif page == "O aplikaci":
    st.title("ℹ️ O aplikaci")
    st.markdown("""
    **Trading Analytics Dashboard** je aplikace pro správu a analýzu pair trading strategií.
    
    ### Hlavní funkce:
    - 📈 Sledování vývoje otevřených pozic
    - 📋 Přehled všech obchodních pozic
    - 🔄 Aktualizace cenových dat
    - ➕ Vytváření a uzavírání pozic
    - 🔍 Vyhledávání pair trading příležitostí
    
    ### Použité technologie:
    - Python
    - Streamlit
    - SQLite
    - Plotly
    - CCXT
    - Statsmodels
    
    ### Autor:
    Aries Jech (2025)
    """)

# Footer
st.markdown("---")
st.caption("Trading Analytics Dashboard © 2025")

# Initialize database
init_db()