import tkinter as tk
from tkinter import messagebox, ttk, filedialog
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import tempfile, webbrowser
from fpdf import FPDF
import os

# --- UI Theme Color Constants ---
BG        = "#0a0a0a"
SURFACE   = "#161616"
FG        = "#ffffff"
MUTED     = "#888888"
BORDER    = "#2e2e2e"
BORDER_H  = "#555555"
BTN_BG    = "#3a1f6e"
BTN_HOV   = "#5b30a8"
BTN_ACT   = "#7c45d4"
ACCENT    = "#c084fc"
GREEN     = "#3ddc84"
RED_C     = "#ff5f5f"
AMBER     = "#ffb347"

FONT      = ("Consolas", 10)
FONT_SM   = ("Consolas", 9)
FONT_LG   = ("Consolas", 13, "bold")

# --- Plotly Specific Theme Constants ---
PLT_BG    = "#0a0a0a"
PLT_SURF  = "#161616"
PLT_GRID  = "#2e2e2e"
PLT_FG    = "#ffffff"
PLT_MUTED = "#888888"
PLT_BLUE  = "#4f9eff"
PLT_GREEN = "#3ddc84"
PLT_AMBER = "#ffb347"
PLT_RED   = "#ff5f5f"
PLT_PURP  = "#c084fc"

def _hover_on(btn, accent_text=False):
    btn.config(bg=BTN_HOV, highlightbackground=BORDER_H,
               fg=ACCENT if accent_text else FG)

def _hover_off(btn):
    btn.config(bg=BTN_BG, highlightbackground="#7c45d4", fg="#ffffff")

def _styled_btn(parent, text, cmd, width=34, accent=False):
    b = tk.Button(parent, text=text, command=cmd,
                  bg=BTN_BG, fg="#ffffff", font=("Consolas", 10, "bold"),
                  relief="raised", width=width, pady=9,
                  anchor="w", padx=12,
                  highlightbackground="#7c45d4",
                  highlightthickness=2,
                  activebackground=BTN_ACT,
                  activeforeground="#ffffff",
                  bd=2,
                  cursor="hand2")
    b.bind("<Enter>", lambda e: _hover_on(b, accent))
    b.bind("<Leave>", lambda e: _hover_off(b))
    return b


class LoginScreen:
    def __init__(self, root):
        self.root = root
        self.root.title("Login")
        self.root.geometry("360x240")
        self.root.configure(bg=BG)
        self.root.resizable(False, False)

        tk.Label(self.root, text="Financial Dashboard",
                 bg=BG, fg=FG, font=FONT_LG).pack(pady=(36, 4))
        tk.Label(self.root, text="Enter password to continue",
                 bg=BG, fg=MUTED, font=FONT_SM).pack()

        self.entry = tk.Entry(self.root, show="●", font=("Consolas", 12),
                              bg=SURFACE, fg=FG, insertbackground=FG,
                              relief="flat",
                              highlightbackground=BORDER,
                              highlightcolor=ACCENT,
                              highlightthickness=1,
                              width=22, justify="center")
        self.entry.pack(pady=16)
        self.entry.focus()

        self.err = tk.Label(self.root, text="", bg=BG, fg=RED_C, font=FONT_SM)
        self.err.pack()

        b = _styled_btn(self.root, "  Unlock", self._check, width=16)
        b.config(anchor="center")
        b.pack(pady=8)
        self.entry.bind("<Return>", lambda _: self._check())

    def _check(self):
        if self.entry.get() == "12345":
            self.root.destroy()
            launch_app()
        else:
            self.entry.delete(0, "end")
            self.err.config(text="Incorrect password. Try again.")


class FinanceApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Financial Analysis Dashboard")
        self.root.geometry("440x760")
        self.root.configure(bg=BG)
        self.root.resizable(False, False)
        self.data = {}
        self.file_path = tk.StringVar(value="No file selected")
        self._selected_path = None
        self._build_ui()

    def _divider(self, parent, pad=28):
        tk.Frame(parent, bg=BORDER, height=1).pack(fill="x", padx=pad)

    def _status(self, msg, colour=MUTED):
        self.root.title(f"Financial Analysis Dashboard  –  {msg}")
        self.root.after(4000, lambda: self.root.title("Financial Analysis Dashboard"))

    def _validate(self):
        if not self.data:
            self._status("⚠  Load data first", RED_C)
            return False
        return True

    def _build_ui(self):
        hdr = tk.Frame(self.root, bg=BG)
        hdr.pack(fill="x", padx=28, pady=(24, 0))
        tk.Label(hdr, text="Financial Analysis Dashboard",
                 bg=BG, fg=FG, font=FONT_LG, anchor="w").pack(side="left")

        tk.Frame(self.root, bg=BG, height=10).pack()
        self._divider(self.root)
        tk.Frame(self.root, bg=BG, height=8).pack()

        row = tk.Frame(self.root, bg=BG)
        row.pack(padx=28, fill="x")
        tk.Label(row, text="File:", bg=BG, fg=MUTED, font=FONT_SM).pack(side="left")
        tk.Label(row, textvariable=self.file_path,
                 bg=BG, fg=FG, font=FONT_SM,
                 width=26, anchor="w").pack(side="left", padx=8)
        b = _styled_btn(row, "Browse", self._browse, width=8)
        b.config(pady=4, anchor="center")
        b.pack(side="left")

        tk.Frame(self.root, bg=BG, height=8).pack()
        self._divider(self.root)
        tk.Frame(self.root, bg=BG, height=6).pack()

        buttons = [
            ("1.  Load Data",             self.load_data),
            ("2.  Clean Data",            self.clean_data),
            ("3.  Structure Analysis",   self.structure_analysis),
            ("4.  KPI Analysis",         self.kpi_analysis),
            ("5.  Trend Analysis",       self.trend_analysis),
            ("6.  Visualizer",           self.visualization),
            ("7.  Statistical Report",   self.statistical_report),
            ("8.  Interactive Dashboard",self.interactive_dashboard),
            ("9.  Advanced Model",       self.advanced_model),
            ("10. Export PDF",           self.export_pdf),
            ("11. Export CSV",           self.export_csv),
            ("12. AI Risk Engine",       self.ai_risk_engine),
        ]

        frame = tk.Frame(self.root, bg=BG)
        frame.pack(padx=28, fill="x")

        for label, cmd in buttons:
            accent = label.startswith("8.") or label.startswith("12.")
            b = _styled_btn(frame, label, cmd, accent=accent)
            b.pack(fill="x", pady=2)

        tk.Frame(self.root, bg=BG, height=8).pack()
        self._divider(self.root)
        tk.Label(self.root, text="Thank You",
                 bg=BG, fg="#222222", font=("Consolas", 8)).pack(pady=(6, 10))

    def _browse(self):
        path = filedialog.askopenfilename(
            title="Select Excel File",
            filetypes=[("Excel Files", "*.xlsx *.xls"), ("All Files", "*.*")])
        if path:
            self._selected_path = path
            self.file_path.set(os.path.basename(path))
            self._status(f"Selected: {os.path.basename(path)}", ACCENT)

    def load_data(self):
        try:
            path = self._selected_path or os.path.join(os.getcwd(), "training_balance.xlsx")
            self.data["2026"] = pd.read_excel(path, sheet_name="BalanceSheet_2026")
            self.data["2027"] = pd.read_excel(path, sheet_name="BalanceSheet_2027")
            self._status("✔  Data loaded successfully", GREEN)
        except Exception as e:
            self._status(f"✘  {e}", RED_C)

    def clean_data(self):
        if not self._validate(): return
        for yr in self.data:
            self.data[yr] = self.data[yr].dropna()
        self._status("✔  Missing values removed", GREEN)

    def structure_analysis(self):
        if not self._validate(): return
        df = self.data["2027"]
        ta = df.loc[df["Item"] == "Total Assets", "Y5"].values[0]
        eq = df.loc[df["Item"] == "Equity",       "Y5"].values[0]
        self._popup("Financial Structure",
                    f"Debt Ratio    {round((ta-eq)/ta, 2)}\n"
                    f"Equity Ratio  {round(eq/ta, 2)}")

    def kpi_analysis(self):
        if not self._validate(): return
        df = self.data["2027"]
        def g(i): return df.loc[df["Item"] == i, "Y5"].values[0]
        self._popup("KPI Analysis",
                    f"Current Ratio      {round(g('Current Assets')/g('Current Liabilities'), 2)}\n"
                    f"Net Profit Margin  {round(g('Net Income')/g('Revenue'), 2)}\n"
                    f"ROA                {round(g('Net Income')/g('Total Assets'), 2)}\n"
                    f"ROE                {round(g('Net Income')/g('Equity'), 2)}")

    def trend_analysis(self):
        if not self._validate(): return
        df = self.data["2027"]
        revenue = df[df["Item"] == "Revenue"].iloc[0, 1:]
        fig, ax = plt.subplots(facecolor=BG)
        ax.set_facecolor(SURFACE)
        ax.plot(revenue, color=ACCENT, linewidth=2, marker="o",
                markersize=6, markerfacecolor=BG, markeredgecolor=ACCENT)
        ax.set_title("Revenue Trend", fontsize=12, color=FG, pad=12)
        ax.set_xlabel("Years (Y0–Y5)", color=MUTED, fontsize=9)
        ax.set_ylabel("Revenue",       color=MUTED, fontsize=9)
        ax.tick_params(colors=MUTED)
        ax.grid(True, color=BORDER, linestyle="--", linewidth=0.8)
        for sp in ax.spines.values(): sp.set_edgecolor(BORDER)
        plt.tight_layout(); plt.show()

    def visualization(self):
        if not self._validate(): return
        df  = self.data["2027"]
        rev = df[df["Item"] == "Revenue"].iloc[0, 1:]
        ni  = df[df["Item"] == "Net Income"].iloc[0, 1:]
        fig, ax = plt.subplots(figsize=(8, 5), facecolor=BG)
        ax.set_facecolor(SURFACE)
        ax.plot(rev, color=ACCENT, lw=2, marker="o", markersize=6,
                markerfacecolor=BG, markeredgecolor=ACCENT, label="Revenue")
        ax.plot(ni,  color=GREEN,  lw=2, marker="s", markersize=6,
                markerfacecolor=BG, markeredgecolor=GREEN,  label="Net Income")
        ax.legend(frameon=False, fontsize=9, labelcolor=FG)
        ax.set_title("Revenue vs Net Income", fontsize=12, color=FG, pad=12)
        ax.tick_params(colors=MUTED)
        ax.grid(True, color=BORDER, linestyle="--", linewidth=0.8)
        for sp in ax.spines.values(): sp.set_edgecolor(BORDER)
        plt.tight_layout(); plt.show()

    def statistical_report(self):
        if not self._validate(): return
        df = self.data["2027"].copy()
        dn = df.set_index("Item").T.apply(pd.to_numeric)
        stats = pd.DataFrame({
            "Mean":     dn.mean(),
            "Std Dev":  dn.std(),
            "Min":      dn.min(),
            "Max":      dn.max(),
            "Growth %": ((dn.iloc[-1] - dn.iloc[0]) / dn.iloc[0]) * 100
        }).round(2)
        stats.reset_index(inplace=True)
        stats.rename(columns={"index": "Item"}, inplace=True)
        self._show_table(stats, "Statistical Report")

    def interactive_dashboard(self):
        if not self._validate(): return
        
        df = self.data["2027"]
        
        # Helper function to get continuous historical row timelines
        def get_timeline(item_name):
            try:
                return df[df["Item"] == item_name].iloc[0, 1:].astype(float)
            except Exception:
                return pd.Series([0.0]*6, index=[f"Y{i}" for i in range(6)])

        # Fetch timelines for calculation
        rev_t  = get_timeline('Revenue')
        ni_t   = get_timeline('Net Income')
        ca_t   = get_timeline('Current Assets')
        cl_t   = get_timeline('Current Liabilities')
        ta_t   = get_timeline('Total Assets')
        eq_t   = get_timeline('Equity')

        # Calculate time-series ratios for the visualization engine
        curr_ratio_t = (ca_t / cl_t).round(2)
        debt_ratio_t = ((ta_t - eq_t) / ta_t).round(2)
        net_margin_t = ((ni_t / rev_t) * 100).round(1)
        roa_t        = ((ni_t / ta_t) * 100).round(1)

        # Multi-row layout schema: Row 1 = 4 KPI metrics, Row 2 = Core interactive line graph canvas
        fig = make_subplots(
            rows=2, cols=4,
            row_heights=[0.22, 0.78],
            specs=[[{"type": "domain"}, {"type": "domain"}, {"type": "domain"}, {"type": "domain"}],
                   [{"colspan": 4}, None, None, None]],
            vertical_spacing=0.1
        )

        # 1. Current Ratio KPI Block
        fig.add_trace(go.Indicator(
            mode="number+delta", value=curr_ratio_t.iloc[-1],
            delta={'reference': curr_ratio_t.iloc[0], 'relative': False, 'valueformat': '.2f'},
            title={"text": "Current Ratio (Y5)<br><span style='font-size:10px;color:#888'>vs Y0 Start</span>", "font": {"size": 12}},
            number={"font": {"color": PLT_BLUE, "size": 26}, "valueformat": ".2f"}
        ), row=1, col=1)

        # 2. Debt Ratio KPI Block
        fig.add_trace(go.Indicator(
            mode="number+delta", value=debt_ratio_t.iloc[-1],
            delta={'reference': debt_ratio_t.iloc[0], 'relative': False, 'valueformat': '.2f', 'increasing': {'color': PLT_RED}, 'decreasing': {'color': PLT_GREEN}},
            title={"text": "Debt Ratio (Y5)<br><span style='font-size:10px;color:#888'>vs Y0 Start</span>", "font": {"size": 12}},
            number={"font": {"color": PLT_RED, "size": 26}, "valueformat": ".2f"}
        ), row=1, col=2)

        # 3. Net Margin KPI Block
        fig.add_trace(go.Indicator(
            mode="number+delta", value=net_margin_t.iloc[-1],
            delta={'reference': net_margin_t.iloc[0], 'relative': True, 'valueformat': '.1%'},
            title={"text": "Net Margin (Y5)<br><span style='font-size:10px;color:#888'>vs Y0 Start</span>", "font": {"size": 12}},
            number={"font": {"color": PLT_GREEN, "size": 26}, "suffix": "%", "valueformat": ".1f"}
        ), row=1, col=3)

        # 4. Return on Assets KPI Block
        fig.add_trace(go.Indicator(
            mode="number+delta", value=roa_t.iloc[-1],
            delta={'reference': roa_t.iloc[0], 'relative': True, 'valueformat': '.1%'},
            title={"text": "Return on Assets (Y5)<br><span style='font-size:10px;color:#888'>vs Y0 Start</span>", "font": {"size": 12}},
            number={"font": {"color": PLT_PURP, "size": 26}, "suffix": "%", "valueformat": ".1f"}
        ), row=1, col=4)

        # Assemble row 2 layers (Initially hidden traces map to interactive dropdown options)
        years = rev_t.index

        # Group View 0: Income Statement (Set Default to visible=True)
        fig.add_trace(go.Scatter(x=years, y=rev_t.values, mode="lines+markers", name="Revenue", visible=True,
                                 line=dict(color=PLT_PURP, width=3.5), marker=dict(size=8), hovertemplate="<b>Revenue</b><br>Year: %{x}<br>Value: %{y:,.0f}<extra></extra>"), row=2, col=1)
        fig.add_trace(go.Scatter(x=years, y=ni_t.values, mode="lines+markers", name="Net Income", visible=True,
                                 line=dict(color=PLT_GREEN, width=3.5), marker=dict(size=8), hovertemplate="<b>Net Income</b><br>Year: %{x}<br>Value: %{y:,.0f}<extra></extra>"), row=2, col=1)

        # Group View 1: Balance Sheet
        fig.add_trace(go.Scatter(x=years, y=ta_t.values, mode="lines+markers", name="Total Assets", visible=False,
                                 line=dict(color=PLT_BLUE, width=3.5), marker=dict(size=8), hovertemplate="<b>Total Assets</b><br>Year: %{x}<br>Value: %{y:,.0f}<extra></extra>"), row=2, col=1)
        fig.add_trace(go.Scatter(x=years, y=eq_t.values, mode="lines+markers", name="Equity", visible=False,
                                 line=dict(color=PLT_AMBER, width=3.5), marker=dict(size=8), hovertemplate="<b>Equity</b><br>Year: %{x}<br>Value: %{y:,.0f}<extra></extra>"), row=2, col=1)

        # Group View 2: Performance Ratio Indicators
        fig.add_trace(go.Scatter(x=years, y=curr_ratio_t.values, mode="lines+markers", name="Current Ratio", visible=False,
                                 line=dict(color=PLT_BLUE, width=2.5, dash='dot'), marker=dict(size=8), hovertemplate="<b>Current Ratio</b><br>Year: %{x}<br>Ratio: %{y:.2f}<extra></extra>"), row=2, col=1)
        fig.add_trace(go.Scatter(x=years, y=net_margin_t.values, mode="lines+markers", name="Net Margin (%)", visible=False,
                                 line=dict(color=PLT_GREEN, width=2.5, dash='dash'), marker=dict(size=8), hovertemplate="<b>Net Margin</b><br>Year: %{x}<br>Margin: %{y}%<extra></extra>"), row=2, col=1)
        fig.add_trace(go.Scatter(x=years, y=roa_t.values, mode="lines+markers", name="ROA (%)", visible=False,
                                 line=dict(color=PLT_PURP, width=2.5, dash='longdash'), marker=dict(size=8), hovertemplate="<b>ROA</b><br>Year: %{x}<br>Rate: %{y}%<extra></extra>"), row=2, col=1)

        # Dynamic Dropdown Menu Filter Configurations
        updatemenus = [
            dict(
                type="dropdown",
                direction="down",
                active=0,
                x=0.0, y=0.78,
                xanchor="left", yanchor="top",
                bgcolor=PLT_SURF,
                bordercolor=BORDER,
                font=dict(family="Consolas", size=12, color=PLT_FG),
                buttons=[
                    dict(
                        label="View: Income Statement Trends",
                        method="update",
                        args=[{"visible": [True, True, True, True, True, True, False, False, False, False, False]},
                              {"yaxis": {"title": "Currency Value (USD)", "gridcolor": PLT_GRID, "zeroline": False}}]
                    ),
                    dict(
                        label="View: Balance Sheet Health",
                        method="update",
                        args=[{"visible": [True, True, True, True, False, False, True, True, False, False, False]},
                              {"yaxis": {"title": "Currency Value (USD)", "gridcolor": PLT_GRID, "zeroline": False}}]
                    ),
                    dict(
                        label="View: Performance Ratios",
                        method="update",
                        args=[{"visible": [True, True, True, True, False, False, False, False, True, True, True]},
                              {"yaxis": {"title": "Ratio Score / Percentage (%)", "gridcolor": PLT_GRID, "zeroline": False}}]
                    )
                ]
            )
        ]

        # Final Dashboard Theme Configurations
        fig.update_layout(
            template="plotly_dark",
            title={"text": "Interactive Executive Financial Command Center", "font": {"size": 18, "family": "Consolas", "color": PLT_FG}, "y": 0.96, "x": 0.01},
            paper_bgcolor=PLT_BG,
            plot_bgcolor=PLT_SURF,
            font={"family": "Consolas", "color": PLT_FG},
            updatemenus=updatemenus,
            legend=dict(orientation="h", yanchor="bottom", y=0.74, xanchor="right", x=1, bgcolor="rgba(0,0,0,0)"),
            margin=dict(l=50, r=40, t=100, b=50),
            height=750
        )
        
        fig.update_xaxes(title_text="Fiscal Period", gridcolor=PLT_GRID, linecolor=PLT_GRID, zeroline=False, row=2, col=1)
        fig.update_yaxes(title_text="Currency Value (USD)", gridcolor=PLT_GRID, linecolor=PLT_GRID, zeroline=False, row=2, col=1)

        # Write to temporary file space and initialize system web browser application
        temp = tempfile.NamedTemporaryFile(delete=False, suffix=".html")
        fig.write_html(temp.name, config={"displaylogo": False, "responsive": True, "displayModeBar": True})
        webbrowser.open("file://" + temp.name)

    def advanced_model(self):
        if not self._validate(): return
        import numpy as np
        df = self.data["2027"].copy()
        dn = df.set_index("Item").T.apply(pd.to_numeric)
        rev = dn["Revenue"]; ni = dn["Net Income"]
        x   = list(range(len(rev)))
        rc  = np.polyfit(x, rev.values, 1)
        nc  = np.polyfit(x, ni.values,  1)
        fut = [len(rev), len(rev)+1, len(rev)+2]
        rp  = [round(np.polyval(rc, f), 0) for f in fut]
        np_ = [round(np.polyval(nc, f), 0) for f in fut]
        lbl = [f"Y{i}" for i in x]
        fl  = [f"Y{i}" for i in fut]
        fig, ax = plt.subplots(figsize=(9, 5), facecolor=BG)
        ax.set_facecolor(SURFACE)
        ax.plot(lbl, rev.values, color=ACCENT, lw=2, marker="o", markersize=5,
                markerfacecolor=BG, markeredgecolor=ACCENT, label="Revenue (actual)")
        ax.plot(lbl, ni.values,  color=GREEN,  lw=2, marker="s", markersize=5,
                markerfacecolor=BG, markeredgecolor=GREEN,  label="Net Income (actual)")
        ax.plot(fl, rp,  color=ACCENT, lw=1.4, ls="--", marker="o", markersize=4,
                markerfacecolor=BG, markeredgecolor=ACCENT, label="Revenue (proj.)")
        ax.plot(fl, np_, color=GREEN,  lw=1.4, ls="--", marker="s", markersize=4,
                markerfacecolor=BG, markeredgecolor=GREEN,  label="Net Income (proj.)")
        ax.axvline(x=lbl[-1], color=BORDER, ls=":", lw=1)
        ax.legend(frameon=False, fontsize=9, labelcolor=FG)
        ax.set_title("Advanced Model  ·  Linear Projection", fontsize=12, color=FG, pad=12)
        ax.tick_params(colors=MUTED)
        ax.grid(True, color=BORDER, ls="--", lw=0.8)
        for sp in ax.spines.values(): sp.set_edgecolor(BORDER)
        plt.tight_layout(); plt.show()

    def export_pdf(self):
        if not self._validate(): return
        try:
            pdf = FPDF(); pdf.add_page(); pdf.set_font("Arial", size=12)
            pdf.cell(200, 10, txt="Financial Summary Report", ln=True)
            df  = self.data["2027"]
            rev = df.loc[df["Item"] == "Revenue",    "Y5"].values[0]
            ni  = df.loc[df["Item"] == "Net Income", "Y5"].values[0]
            pdf.cell(200, 10, txt=f"Revenue (Y5):    {rev}", ln=True)
            pdf.cell(200, 10, txt=f"Net Income (Y5): {ni}",  ln=True)
            pdf.output("Financial_Report.pdf")
            self._status("✔  PDF exported", GREEN)
        except Exception as e:
            self._status(f"✘  {e}", RED_C)

    def export_csv(self):
        if not self._validate(): return
        self.data["2027"].to_csv("Financial_Data_2027.csv", index=False)
        self._status("✔  CSV exported", GREEN)

    def ai_risk_engine(self):
        if not self._validate(): return
        try:
            df = self.data["2027"]
            def g(i): return float(df.loc[df["Item"] == i, "Y5"].values[0])
            ta  = g("Total Assets");       eq = g("Equity")
            rev = g("Revenue");             ni = g("Net Income")
            ca  = g("Current Assets");      cl = g("Current Liabilities")
            cr  = ca / cl
            dr  = (ta - eq) / ta
            pm  = ni / rev
            roa = ni / ta
            score = round(
                (1 - min(cr / 2, 1)) * 25 +
                dr * 35 +
                (1 - max(pm, 0)) * 20 +
                (1 - max(roa * 10, 0)) * 20, 2)
            if score < 30:
                lvl = "LOW ✔";    col = GREEN; rec = "Strong financial health.\nStable liquidity and profitability."
            elif score < 60:
                lvl = "MODERATE"; col = AMBER; rec = "Moderate risk.\nMonitor debt and improve margins."
            else:
                lvl = "HIGH ✘";   col = RED_C; rec = "High risk.\nFocus on liquidity and debt reduction."
            self._popup("AI Risk Engine",
                        f"Risk Score   {score} / 100\n"
                        f"Risk Level   {lvl}\n\n"
                        f"{rec}", colour=col)
        except Exception as e:
            self._status(f"✘  {e}", RED_C)

    def _popup(self, title, body, colour=FG):
        win = tk.Toplevel(self.root)
        win.title(title)
        win.configure(bg=BG)
        win.resizable(False, False)
        win.grab_set()
        tk.Label(win, text=title, bg=BG, fg=FG,
                 font=("Consolas", 12, "bold")).pack(padx=30, pady=(20, 6))
        self._divider(win, pad=20)
        tk.Label(win, text=body, bg=BG, fg=colour,
                 font=("Consolas", 10), justify="left").pack(padx=30, pady=16)
        self._divider(win, pad=20)
        b = _styled_btn(win, "  Close", win.destroy, width=10)
        b.config(anchor="center")
        b.pack(pady=12)

    def _show_table(self, df, title):
        win = tk.Toplevel(self.root)
        win.title(title)
        win.geometry("1000x500")
        win.configure(bg=BG)
        style = ttk.Style(win)
        style.theme_use("clam")
        style.configure("Blk.Treeview",
                        background=SURFACE, foreground=FG, rowheight=26,
                        fieldbackground=SURFACE, font=("Consolas", 9))
        style.configure("Blk.Treeview.Heading",
                        background=BTN_BG, foreground=ACCENT, relief="flat",
                        font=("Consolas", 9, "bold"))
        style.map("Blk.Treeview",
                  background=[("selected", BTN_HOV)],
                  foreground=[("selected", ACCENT)])
        tree = ttk.Treeview(win, style="Blk.Treeview")
        tree.pack(fill="both", expand=True, padx=16, pady=16)
        tree["columns"] = list(df.columns)
        tree["show"]    = "headings"
        for col in df.columns:
            tree.heading(col, text=col)
            tree.column(col, width=130, anchor="center")
        for _, row in df.iterrows():
            tree.insert("", "end", values=list(row))


def launch_app():
    r = tk.Tk()
    FinanceApp(r)
    r.mainloop()

login_root = tk.Tk()
LoginScreen(login_root)
login_root.mainloop()
