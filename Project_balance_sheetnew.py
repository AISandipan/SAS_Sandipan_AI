import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import pandas as pd
import matplotlib.pyplot as plt

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from fpdf import FPDF

import tempfile
import webbrowser


class FinanceAnalyticsApp:

    def __init__(self, root):

        self.root = root
        self.root.title("Finance + Sales Analytics Platform")
        self.root.geometry("650x900")

        self.data = {}

        self.dataset_type = tk.StringVar(
            value="Financial"
        )

        #########################################

        tk.Label(

            root,

            text="Select Dataset Type",

            font=("Arial",12,"bold")

        ).pack()

        tk.Radiobutton(

            root,

            text="Financial Statements",

            variable=self.dataset_type,

            value="Financial"

        ).pack()

        tk.Radiobutton(

            root,

            text="Sales Dataset",

            variable=self.dataset_type,

            value="Sales"

        ).pack()

        #########################################

        buttons = [

            ("1. Load Data", self.load_data),

            ("2. Clean Data", self.clean_data),

            ("3. Structure Analysis", self.structure_analysis),

            ("4. Ratio / KPI Analysis", self.ratio_analysis),

            ("5. Trend Analysis", self.trend_analysis),

            ("6. Visualize", self.visualization),

            ("7. Statistical Report", self.statistical_report),

            ("8. Interactive Dashboard", self.interactive_dashboard),

            ("9. Advanced Model", self.advanced_model),

            ("10. Export PDF", self.export_pdf),

            ("11. Export CSV", self.export_csv)

        ]

        for txt,cmd in buttons:

            tk.Button(

                root,

                text=txt,

                width=35,

                command=cmd

            ).pack(pady=4)

    ###################################################

    def validate(self):

        if len(self.data)==0:

            messagebox.showwarning(

                "Warning",

                "Load data first"

            )

            return False

        return True

    ###################################################

    def load_data(self):

        try:

            file = filedialog.askopenfilename(

                filetypes=[

                    ("Excel","*.xlsx"),

                    ("CSV","*.csv")

                ]

            )

            if not file:

                return

            mode = self.dataset_type.get()

            if file.endswith(".csv"):

                self.data["main"] = pd.read_csv(file)

            else:

                if mode=="Financial":

                    self.data["2026"] = pd.read_excel(

                        file,

                        sheet_name="BalanceSheet_2026"

                    )

                    self.data["2027"] = pd.read_excel(

                        file,

                        sheet_name="BalanceSheet_2027"

                    )

                else:

                    self.data["main"] = pd.read_excel(file)

            messagebox.showinfo(

                "Success",

                "Dataset Loaded"

            )

        except Exception as e:

            messagebox.showerror(

                "Error",

                str(e)

            )

    ###################################################

    def clean_data(self):

        if not self.validate():
            return

        for k in self.data:

            self.data[k] = self.data[k].dropna()

        messagebox.showinfo(

            "Done",

            "Missing Values Removed"

        )

    ###################################################

    def structure_analysis(self):

        if not self.validate():
            return

        mode = self.dataset_type.get()

        if mode=="Financial":

            df = self.data["2027"]

            assets = df.loc[
                df["Item"]=="Total Assets",
                "Y5"
            ].values[0]

            equity = df.loc[
                df["Item"]=="Equity",
                "Y5"
            ].values[0]

            debt_ratio = (

                assets-equity

            ) / assets

            messagebox.showinfo(

                "Structure",

                f"Debt Ratio = {round(debt_ratio,2)}"

            )

        else:

            df = self.data["main"]

            region_sales = df.groupby(

                "Region"

            )["Sales"].sum()

            top = region_sales.idxmax()

            messagebox.showinfo(

                "Top Region",

                top

            )

    ###################################################

    def ratio_analysis(self):

        if not self.validate():
            return

        mode = self.dataset_type.get()

        if mode=="Financial":

            df = self.data["2027"]

            revenue = df.loc[
                df["Item"]=="Revenue",
                "Y5"
            ].values[0]

            income = df.loc[
                df["Item"]=="Net Income",
                "Y5"
            ].values[0]

            margin = income/revenue

            messagebox.showinfo(

                "Financial Ratio",

                f"Profit Margin = {round(margin,2)}"

            )

        else:

            df = self.data["main"]

            sales = df["Sales"].sum()

            profit = df["Profit"].sum()

            margin = (

                profit/sales

            ) * 100

            avg_order = sales/len(df)

            msg = f"""

Sales = {round(sales,2)}

Profit = {round(profit,2)}

Margin = {round(margin,2)}%

Average Order = {round(avg_order,2)}

"""

            messagebox.showinfo(

                "Sales KPI",

                msg

            )

    ###################################################

    def trend_analysis(self):

        if not self.validate():
            return

        mode = self.dataset_type.get()

        plt.figure(figsize=(8,5))

        if mode=="Financial":

            revenue = self.data["2027"][

                self.data["2027"]["Item"]

                =="Revenue"

            ].iloc[0,1:]

            plt.plot(

                revenue.index,

                revenue.values,

                marker="o"

            )

            plt.title(

                "Revenue Trend"

            )

        else:

            df = self.data["main"]

            df["Order Date"] = pd.to_datetime(

                df["Order Date"]

            )

            monthly = df.groupby(

                df["Order Date"]

                .dt.to_period("M")

            )["Sales"].sum()

            plt.plot(

                monthly.index.astype(str),

                monthly.values,

                marker="o"

            )

            plt.title(

                "Monthly Sales Trend"

            )

        plt.xticks(rotation=45)

        plt.grid(True)

        plt.show()

    ###################################################

    def visualization(self):

        self.interactive_dashboard()

    ###################################################

    def interactive_dashboard(self):

        if not self.validate():
            return

        mode = self.dataset_type.get()

        fig = make_subplots(

            rows=1,

            cols=1

        )

        if mode=="Financial":

            df = self.data["2027"]

            revenue = df[

                df["Item"]=="Revenue"

            ].iloc[0,1:]

            fig.add_trace(

                go.Scatter(

                    x=revenue.index,

                    y=revenue.values,

                    mode="lines+markers",

                    name="Revenue"

                )

            )

        else:

            df = self.data["main"]

            category = df.groupby(

                "Category"

            )["Sales"].sum()

            fig.add_trace(

                go.Bar(

                    x=category.index,

                    y=category.values,

                    name="Sales"

                )

            )

        fig.update_layout(

            title="Interactive Dashboard",

            height=600

        )

        temp = tempfile.NamedTemporaryFile(

            delete=False,

            suffix=".html"

        )

        fig.write_html(

            temp.name

        )

        webbrowser.open(

            "file://" + temp.name

        )

    ###################################################

    def advanced_model(self):

        if not self.validate():
            return

        mode = self.dataset_type.get()

        if mode=="Financial":

            df = self.data["2027"]

            y0 = df.loc[
                df["Item"]=="Revenue",
                "Y0"
            ].values[0]

            y5 = df.loc[
                df["Item"]=="Revenue",
                "Y5"
            ].values[0]

            cagr = (

                (y5/y0)**(1/5)-1

            )*100

            messagebox.showinfo(

                "Financial Model",

                f"CAGR={round(cagr,2)}%"

            )

        else:

            df = self.data["main"]

            corr = df["Sales"].corr(

                df["Profit"]

            )

            messagebox.showinfo(

                "Correlation",

                round(corr,2)

            )

    ###################################################

    def statistical_report(self):

        if not self.validate():
            return

        df = list(

            self.data.values()

        )[0]

        stats = df.describe()

        self.show_table(stats)

    ###################################################

    def show_table(self, dataframe):

        win = tk.Toplevel()

        tree = ttk.Treeview(win)

        tree.pack(

            fill="both",

            expand=True

        )

        cols = ["Index"] + list(dataframe.columns)

        tree["columns"] = cols

        tree["show"] = "headings"

        for c in cols:

            tree.heading(

                c,

                text=c

            )

        for idx,row in dataframe.iterrows():

            tree.insert(

                "",

                "end",

                values=[idx]+list(row)

            )

    ###################################################

    def export_pdf(self):

        pdf = FPDF()

        pdf.add_page()

        pdf.set_font(

            "Arial",

            size=12

        )

        pdf.cell(

            200,

            10,

            txt="Analytics Report",

            ln=True

        )

        pdf.output(

            "Analytics_Report.pdf"

        )

        messagebox.showinfo(

            "Done",

            "PDF Exported"

        )

    ###################################################

    def export_csv(self):

        df = list(

            self.data.values()

        )[0]

        df.to_csv(

            "Exported_Data.csv",

            index=False

        )

        messagebox.showinfo(

            "Done",

            "CSV Exported"

        )


########################################################

root = tk.Tk()

app = FinanceAnalyticsApp(root)

root.mainloop()