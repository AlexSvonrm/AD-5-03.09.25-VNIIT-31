# -*- coding: utf-8 -*-
"""Анализ продаж: от ETL до инсайтов"""

# 1. ИМПОРТ БИБЛИОТЕК
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Визуализация
import plotly.express as px
import plotly.graph_objects as go
import plotly.subplots as sp
from plotly.offline import plot

# Машинное обучение (для кластеризации в RFM)
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Настройки отображения
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# 2. КОНФИГУРАЦИЯ
FILE_PATHS = {
    'customer': 'Data/Customer.xlsx',
    'product': 'Data/Product.xlsx',
    'sales': 'Data/Sales.xlsx',
    'territories': 'Data/Territories.xlsx'
}

# 3. ФУНКЦИИ ETL
def load_data():
    """Загрузка всех исходных файлов"""
    try:
        customer_df = pd.read_excel(FILE_PATHS['customer'])
        product_df = pd.read_excel(FILE_PATHS['product'])
        sales_df = pd.read_excel(FILE_PATHS['sales'])
        territories_df = pd.read_excel(FILE_PATHS['territories'])

        print("✅ Данные успешно загружены")
        print(f"📊 Размеры данных:")
        print(f"   - Customers: {customer_df.shape}")
        print(f"   - Products: {product_df.shape}")
        print(f"   - Sales: {sales_df.shape}")
        print(f"   - Territories: {territories_df.shape}")

        return customer_df, product_df, sales_df, territories_df

    except Exception as e:
        print(f"❌ Ошибка загрузки: {e}")
        return None, None, None, None

def merge_data(customer_df, product_df, sales_df, territories_df):
    """Объединение таблиц в единый датафрейм"""

    # Основное объединение
    merged_df = sales_df.merge(
        customer_df, on='CustomerKey', how='left', suffixes=('', '_customer')
    ).merge(
        product_df, on='ProductKey', how='left', suffixes=('', '_product')
    ).merge(
        territories_df, left_on='SalesTerritoryKey', right_on='SalesTerritoryKey',
        how='left', suffixes=('', '_territory')
    )

    print("✅ Данные объединены")
    print(f"📊 Размер объединенного датасета: {merged_df.shape}")

    return merged_df

def clean_data(df):
    """Очистка и преобразование данных"""

    print("🧹 ОЧИСТКА ДАННЫХ")
    print("=" * 40)

    # Сохраняем исходный размер
    initial_count = len(df)
    print(f"📊 Исходное количество строк: {initial_count}")

    # Удаление дубликатов
    df = df.drop_duplicates()
    print(f"📊 Удалено дубликатов: {initial_count - len(df)}")

    # Анализ пропущенных значений
    missing_data = df.isnull().sum()
    missing_percent = (missing_data / len(df)) * 100

    print("\n📊 Пропущенные значения:")
    missing_info = pd.DataFrame({
        'Количество': missing_data,
        'Процент': missing_percent
    }).sort_values('Количество', ascending=False)

    # Показываем только колонки с пропусками
    missing_columns = missing_info[missing_info['Количество'] > 0]
    if len(missing_columns) > 0:
        display(missing_columns)
    else:
        print("   Пропущенных значений не обнаружено")

    # Заполнение пропусков для ключевых колонок
    if 'Color' in df.columns:
        df['Color'] = df['Color'].fillna('Not Specified')

    if 'SubCategory' in df.columns:
        df['SubCategory'] = df['SubCategory'].fillna('Unknown')
        df['Category'] = df['Category'].fillna('Unknown')

    # Преобразование типов данных
    date_columns = ['OrderDate', 'ShipDate', 'BirthDate', 'DateFirstPurchase', 'StartDate']

    for col in date_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
            print(f"🕐 Преобразована колонка: {col}")

    # Преобразование числовых колонок
    numeric_columns = ['YearlyIncome', 'TotalChildren', 'NumberChildrenAtHome',
                      'NumberCarsOwned', 'StandardCost', 'ListPrice', 'DaysToManufacture',
                      'OrderQuantity', 'UnitPrice', 'TotalProductCost', 'SalesAmount', 'TaxAmt']

    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            print(f"🔢 Преобразована числовая колонка: {col}")

    # Создание новых признаков
    if 'OrderDate' in df.columns:
        df['OrderYear'] = df['OrderDate'].dt.year
        df['OrderMonth'] = df['OrderDate'].dt.month
        df['OrderQuarter'] = df['OrderDate'].dt.quarter
        df['OrderDayOfWeek'] = df['OrderDate'].dt.day_name()

        # Расчет времени доставки
        if 'ShipDate' in df.columns:
            df['DeliveryDays'] = (df['ShipDate'] - df['OrderDate']).dt.days

    # Создание демографических сегментов
    if 'YearlyIncome' in df.columns:
        df['IncomeSegment'] = pd.cut(df['YearlyIncome'],
                                   bins=[0, 50000, 80000, 100000, float('inf')],
                                   labels=['Low', 'Medium', 'High', 'Very High'])

    # Расчет возраста клиента
    if 'BirthDate' in df.columns:
        df['Age'] = (datetime.now() - df['BirthDate']).dt.days // 365

    # Расчет прибыльности
    if 'SalesAmount' in df.columns and 'TotalProductCost' in df.columns:
        df['Profit'] = df['SalesAmount'] - df['TotalProductCost']
        df['ProfitMargin'] = (df['Profit'] / df['SalesAmount']) * 100

    print("✅ Данные очищены и преобразованы")
    return df

def perform_eda(df):
    """Исследовательский анализ данных"""

    print("🔍 ИССЛЕДОВАТЕЛЬСКИЙ АНАЛИЗ ДАННЫХ")
    print("=" * 50)

    # Базовая статистика числовых колонок
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    print("📈 ОСНОВНАЯ СТАТИСТИКА (числовые колонки):")
    display(df[numeric_cols].describe())

    # Анализ категориальных переменных
    categorical_cols = df.select_dtypes(include=['object']).columns
    print(f"\n📊 КАТЕГОРИАЛЬНЫЕ ПЕРЕМЕННЫЕ ({len(categorical_cols)}):")

    for col in categorical_cols[:8]:  # Показываем первые 8
        unique_count = df[col].nunique()
        print(f"   {col}: {unique_count} уникальных значений")
        if unique_count <= 10:
            value_counts = df[col].value_counts()
            print(f"      Распределение: {value_counts.to_dict()}")

    return df

def create_eda_visualizations(df):
    """Создание визуализаций для EDA"""

    print("📊 СОЗДАНИЕ ВИЗУАЛИЗАЦИЙ ДЛЯ EDA")

    # Создаем subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # 1. Динамика продаж по времени
    if 'OrderDate' in df.columns and 'SalesAmount' in df.columns:
        monthly_sales = df.groupby(df['OrderDate'].dt.to_period('M'))['SalesAmount'].sum()
        axes[0,0].plot(monthly_sales.index.astype(str), monthly_sales.values,
                      marker='o', linewidth=2, color='blue')
        axes[0,0].set_title('Динамика продаж по месяцам', fontsize=14, fontweight='bold')
        axes[0,0].set_xlabel('Месяц')
        axes[0,0].set_ylabel('Выручка (SalesAmount)')
        axes[0,0].tick_params(axis='x', rotation=45)
        axes[0,0].grid(True, alpha=0.3)

    # 2. Распределение по регионам
    if 'Region' in df.columns:
        region_sales = df.groupby('Region')['SalesAmount'].sum().sort_values(ascending=False)
        axes[0,1].bar(range(len(region_sales)), region_sales.values, color='green', alpha=0.7)
        axes[0,1].set_title('Выручка по регионам', fontsize=14, fontweight='bold')
        axes[0,1].set_xlabel('Регион')
        axes[0,1].set_ylabel('Выручка')
        axes[0,1].set_xticks(range(len(region_sales)))
        axes[0,1].set_xticklabels(region_sales.index, rotation=45)

    # 3. Распределение по категориям продуктов
    if 'Category' in df.columns:
        category_sales = df.groupby('Category')['SalesAmount'].sum()
        axes[0,2].pie(category_sales.values, labels=category_sales.index,
                     autopct='%1.1f%%', startangle=90)
        axes[0,2].set_title('Распределение выручки по категориям', fontsize=14, fontweight='bold')

    # 4. Распределение доходов клиентов
    if 'YearlyIncome' in df.columns:
        axes[1,0].hist(df['YearlyIncome'], bins=20, alpha=0.7, edgecolor='black', color='orange')
        axes[1,0].set_title('Распределение годового дохода клиентов', fontsize=14, fontweight='bold')
        axes[1,0].set_xlabel('Годовой доход')
        axes[1,0].set_ylabel('Количество клиентов')

    # 5. Распределение суммы продаж
    if 'SalesAmount' in df.columns:
        axes[1,1].hist(df['SalesAmount'], bins=50, alpha=0.7, edgecolor='black', color='red')
        axes[1,1].set_title('Распределение суммы продаж', fontsize=14, fontweight='bold')
        axes[1,1].set_xlabel('Сумма продажи')
        axes[1,1].set_ylabel('Количество')

    # 6. Топ продуктов по выручке
    if 'ProductName' in df.columns:
        top_products = df.groupby('ProductName')['SalesAmount'].sum().nlargest(10)
        axes[1,2].barh(range(len(top_products)), top_products.values, color='purple', alpha=0.7)
        axes[1,2].set_yticks(range(len(top_products)))
        axes[1,2].set_yticklabels([label[:25] + '...' if len(label) > 25 else label
                                 for label in top_products.index])
        axes[1,2].set_title('Топ-10 продуктов по выручке', fontsize=14, fontweight='bold')
        axes[1,2].set_xlabel('Выручка')

    plt.tight_layout()
    plt.savefig('eda_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Дополнительные интерактивные визуализации с Plotly
    if 'OrderDate' in df.columns and 'SalesAmount' in df.columns:
        # Ежедневные продажи
        daily_sales = df.groupby(df['OrderDate'].dt.date)['SalesAmount'].sum().reset_index()
        fig1 = px.line(daily_sales, x='OrderDate', y='SalesAmount',
                      title='📈 Динамика ежедневных продаж',
                      labels={'OrderDate': 'Дата', 'SalesAmount': 'Выручка'})
        fig1.show()

    if 'Region' in df.columns and 'SalesAmount' in df.columns:
        # Продажи по регионам
        region_performance = df.groupby('Region').agg({
            'SalesAmount': 'sum',
            'CustomerKey': 'nunique',
            'OrderQuantity': 'sum'
        }).reset_index()

        fig2 = px.bar(region_performance, x='Region', y='SalesAmount',
                     title='🌍 Выручка по регионам',
                     color='CustomerKey',
                     labels={'SalesAmount': 'Выручка', 'CustomerKey': 'Количество клиентов'})
        fig2.show()

def save_cleaned_data(df):
    """Сохранение очищенного датасета"""
    df.to_csv('cleaned_sales_data.csv', index=False, encoding='utf-8')
    print("✅ Очищенные данные сохранены в 'cleaned_sales_data.csv'")

# 4. ФУНКЦИИ АНАЛИЗА И ВИЗУАЛИЗАЦИИ
def create_dashboards(df):
    """Создание интерактивных дашбордов"""

    print("📊 СОЗДАНИЕ ДАШБОРДОВ")

    # 1. Дашборд ключевых метрик
    total_revenue = df['SalesAmount'].sum()
    total_orders = df['SalesOrderNumber'].nunique()
    total_customers = df['CustomerKey'].nunique()
    avg_order_value = df.groupby('SalesOrderNumber')['SalesAmount'].sum().mean()
    total_profit = df['Profit'].sum() if 'Profit' in df.columns else 0

    # Создаем дашборд с метриками
    fig_metrics = go.Figure()

    fig_metrics.add_trace(go.Indicator(
        mode="number",
        value=total_revenue,
        title={"text": "💰 Общая выручка"},
        number={'prefix': "$", "valueformat": ",.0f"},
        domain={'row': 0, 'column': 0}
    ))

    fig_metrics.add_trace(go.Indicator(
        mode="number",
        value=total_orders,
        title={"text": "📦 Всего заказов"},
        number={"valueformat": ",.0f"},
        domain={'row': 0, 'column': 1}
    ))

    fig_metrics.add_trace(go.Indicator(
        mode="number",
        value=total_customers,
        title={"text": "👥 Всего клиентов"},
        number={"valueformat": ",.0f"},
        domain={'row': 0, 'column': 2}
    ))

    fig_metrics.add_trace(go.Indicator(
        mode="number",
        value=avg_order_value,
        title={"text": "💵 Средний чек"},
        number={'prefix': "$", "valueformat": ",.0f"},
        domain={'row': 0, 'column': 3}
    ))

    fig_metrics.update_layout(
        grid={'rows': 1, 'columns': 4, 'pattern': "independent"},
        template="plotly_white",
        height=300
    )

    fig_metrics.show()

    # 2. Дашборд географического распределения
    if 'Country' in df.columns and 'Region' in df.columns:
        geo_analysis = df.groupby(['Country', 'Region']).agg({
            'SalesAmount': 'sum',
            'CustomerKey': 'nunique',
            'Profit': 'sum'
        }).reset_index()

        fig_geo = px.scatter_geo(geo_analysis,
                                locations='Country',
                                locationmode='country names',
                                color='SalesAmount',
                                size='CustomerKey',
                                hover_name='Region',
                                hover_data={'SalesAmount': ':.0f', 'Profit': ':.0f'},
                                title='🌍 Географическое распределение продаж')
        fig_geo.show()

    # 3. Анализ продуктовой линейки
    if 'Category' in df.columns and 'SubCategory' in df.columns:
        product_analysis = df.groupby(['Category', 'SubCategory']).agg({
            'SalesAmount': 'sum',
            'Profit': 'sum',
            'OrderQuantity': 'sum'
        }).reset_index()

        fig_products = px.treemap(product_analysis,
                                 path=['Category', 'SubCategory'],
                                 values='SalesAmount',
                                 color='Profit',
                                 color_continuous_scale='RdYlGn',
                                 title='📦 Дерево продуктов по выручке и прибыли')
        fig_products.show()

    print("✅ Дашборды созданы и отображены")

def extract_insights(df):
    """Извлечение ключевых инсайтов"""

    print("💡 КЛЮЧЕВЫЕ ИНСАЙТЫ")
    print("=" * 50)

    insights = []

    # 1. Общие метрики бизнеса
    total_revenue = df['SalesAmount'].sum()
    total_customers = df['CustomerKey'].nunique()
    total_products = df['ProductKey'].nunique()
    avg_order_value = df.groupby('SalesOrderNumber')['SalesAmount'].sum().mean()

    insights.append(f"💰 Общая выручка: ${total_revenue:,.0f}")
    insights.append(f"👥 Количество клиентов: {total_customers}")
    insights.append(f"📦 Количество продуктов: {total_products}")
    insights.append(f"📊 Средний чек: ${avg_order_value:,.0f}")

    # 2. Анализ клиентской базы
    customer_value = df.groupby('CustomerKey')['SalesAmount'].sum()
    if len(customer_value) > 0:
        top_20_percent = int(len(customer_value) * 0.2)
        top_20_customers = customer_value.nlargest(top_20_percent).sum()
        percent_top_20 = (top_20_customers / total_revenue) * 100

        insights.append(f"🎯 Топ-20% клиентов приносят {percent_top_20:.1f}% выручки")

    # 3. Анализ регионов
    if 'Region' in df.columns:
        best_region = df.groupby('Region')['SalesAmount'].sum().idxmax()
        best_region_revenue = df.groupby('Region')['SalesAmount'].sum().max()
        insights.append(f"🌍 Лучший регион: {best_region} (${best_region_revenue:,.0f})")

    # 4. Анализ продуктов
    if 'Category' in df.columns:
        best_category = df.groupby('Category')['SalesAmount'].sum().idxmax()
        best_category_revenue = df.groupby('Category')['SalesAmount'].sum().max()
        insights.append(f"📦 Лучшая категория: {best_category} (${best_category_revenue:,.0f})")

    # 5. Временной анализ
    if 'OrderDate' in df.columns:
        monthly_revenue = df.groupby(df['OrderDate'].dt.month)['SalesAmount'].sum()
        if len(monthly_revenue) > 0:
            best_month = monthly_revenue.idxmax()
            worst_month = monthly_revenue.idxmin()

            month_names = {
                1: 'Январь', 2: 'Февраль', 3: 'Март', 4: 'Апрель',
                5: 'Май', 6: 'Июнь', 7: 'Июль', 8: 'Август',
                9: 'Сентябрь', 10: 'Октябрь', 11: 'Ноябрь', 12: 'Декабрь'
            }

            insights.append(f"📈 Лучший месяц: {month_names.get(best_month, best_month)} (${monthly_revenue[best_month]:,.0f})")
            insights.append(f"📉 Худший месяц: {month_names.get(worst_month, worst_month)} (${monthly_revenue[worst_month]:,.0f})")

    # 6. Анализ прибыльности
    if 'Profit' in df.columns:
        total_profit = df['Profit'].sum()
        avg_profit_margin = (total_profit / total_revenue) * 100
        insights.append(f"💵 Общая прибыль: ${total_profit:,.0f}")
        insights.append(f"📊 Средняя маржа прибыли: {avg_profit_margin:.1f}%")

    print("🎯 ОСНОВНЫЕ ИНСАЙТЫ:")
    for i, insight in enumerate(insights, 1):
        print(f"{i}. {insight}")

    return insights

# 5. ФУНКЦИИ ОБОГАЩЕНИЯ ДАННЫХ
def perform_rfm_analysis(df):
    """Проведение RFM-анализа"""

    print("🎯 RFM-АНАЛИЗ")
    print("=" * 30)

    # Расчет RFM метрик
    snapshot_date = df['OrderDate'].max() + pd.Timedelta(days=1)

    rfm = df.groupby('CustomerKey').agg({
        'OrderDate': lambda x: (snapshot_date - x.max()).days,  # Recency
        'SalesOrderNumber': 'nunique',  # Frequency
        'SalesAmount': 'sum'            # Monetary
    }).reset_index()

    rfm.columns = ['CustomerKey', 'Recency', 'Frequency', 'Monetary']

    print("📊 Основные статистики RFM:")
    display(rfm[['Recency', 'Frequency', 'Monetary']].describe())

    # Создание RFM квантилей
    rfm['R_Score'] = pd.qcut(rfm['Recency'], 5, labels=[5, 4, 3, 2, 1])
    rfm['F_Score'] = pd.qcut(rfm['Frequency'].rank(method='first'), 5, labels=[1, 2, 3, 4, 5])
    rfm['M_Score'] = pd.qcut(rfm['Monetary'].rank(method='first'), 5, labels=[1, 2, 3, 4, 5])

    # RFM сегментация
    rfm['RFM_Score'] = rfm['R_Score'].astype(str) + rfm['F_Score'].astype(str) + rfm['M_Score'].astype(str)

    # Упрощенная сегментация
    def assign_segment(row):
        r, f, m = int(row['R_Score']), int(row['F_Score']), int(row['M_Score'])

        if r >= 4 and f >= 4 and m >= 4:
            return 'Champions'
        elif r >= 3 and f >= 3 and m >= 3:
            return 'Loyal Customers'
        elif r >= 4 and f <= 2 and m <= 2:
            return 'New Customers'
        elif r >= 3 and f <= 2 and m <= 2:
            return 'Promising'
        elif r <= 2 and f >= 3 and m >= 3:
            return 'Need Attention'
        elif r <= 2 and f <= 2 and m <= 2:
            return 'Hibernating'
        else:
            return 'Regular'

    rfm['Segment'] = rfm.apply(assign_segment, axis=1)

    # Визуализация RFM
    fig = px.scatter(rfm, x='Recency', y='Monetary', color='Segment',
                    size='Frequency', hover_data=['CustomerKey'],
                    title='RFM-анализ клиентов',
                    labels={'Recency': 'Дней с последнего заказа (Recency)',
                           'Monetary': 'Общая выручка (Monetary)'})
    fig.show()

    # Распределение сегментов
    segment_distribution = rfm['Segment'].value_counts()
    print("\n📊 РАСПРЕДЕЛЕНИЕ СЕГМЕНТОВ:")
    for segment, count in segment_distribution.items():
        percentage = (count / len(rfm)) * 100
        print(f"   {segment}: {count} клиентов ({percentage:.1f}%)")

    # Визуализация распределения сегментов
    fig_pie = px.pie(segment_distribution,
                     values=segment_distribution.values,
                     names=segment_distribution.index,
                     title='Распределение клиентов по RFM-сегментам')
    fig_pie.show()

    return rfm

def perform_abc_xyz_analysis(df):
    """Проведение ABC-XYZ анализа продуктов"""

    print("📦 ABC-XYZ АНАЛИЗ ПРОДУКТОВ")
    print("=" * 40)

    # ABC анализ (по выручке)
    product_sales = df.groupby(['ProductKey', 'ProductName']).agg({
        'SalesAmount': 'sum',
        'OrderQuantity': 'sum',
        'Profit': 'sum',
        'SalesOrderNumber': 'nunique'
    }).reset_index()

    product_sales = product_sales.sort_values('SalesAmount', ascending=False)
    product_sales['CumulativePercentage'] = (product_sales['SalesAmount'].cumsum() /
                                           product_sales['SalesAmount'].sum() * 100)

    # ABC категории
    def abc_category(x):
        if x <= 80:
            return 'A'
        elif x <= 95:
            return 'B'
        else:
            return 'C'

    product_sales['ABC_Category'] = product_sales['CumulativePercentage'].apply(abc_category)

    # XYZ анализ (по стабильности продаж)
    try:
        monthly_product_sales = df.pivot_table(
            index='ProductKey',
            columns=df['OrderDate'].dt.to_period('M'),
            values='OrderQuantity',
            aggfunc='sum'
        ).fillna(0)

        # Коэффициент вариации
        cv = monthly_product_sales.std(axis=1) / (monthly_product_sales.mean(axis=1) + 0.0001)

        product_sales = product_sales.merge(cv.rename('CV'), left_on='ProductKey', right_index=True)

        # XYZ категории
        def xyz_category(cv):
            if cv <= 0.3:
                return 'X'
            elif cv <= 0.6:
                return 'Y'
            else:
                return 'Z'

        product_sales['XYZ_Category'] = product_sales['CV'].apply(xyz_category)
        product_sales['ABC_XYZ'] = product_sales['ABC_Category'] + '-' + product_sales['XYZ_Category']

    except Exception as e:
        print(f"⚠️ XYZ анализ не удался: {e}")
        product_sales['XYZ_Category'] = 'Unknown'
        product_sales['ABC_XYZ'] = product_sales['ABC_Category'] + '-Unknown'

    # Визуализация ABC-XYZ матрицы
    if 'CV' in product_sales.columns:
        fig = px.scatter(product_sales, x='CV', y='SalesAmount', color='ABC_XYZ',
                        size='OrderQuantity', hover_data=['ProductName'],
                        title='ABC-XYZ анализ продуктов',
                        labels={'CV': 'Коэффициент вариации (стабильность)',
                               'SalesAmount': 'Выручка',
                               'OrderQuantity': 'Количество проданных единиц'})
        fig.show()

    print("📊 РАСПРЕДЕЛЕНИЕ ABC-КАТЕГОРИЙ:")
    abc_counts = product_sales['ABC_Category'].value_counts()
    for category, count in abc_counts.items():
        percentage = (count / len(product_sales)) * 100
        revenue_share = (product_sales[product_sales['ABC_Category'] == category]['SalesAmount'].sum() /
                        product_sales['SalesAmount'].sum() * 100)
        print(f"   {category}: {count} продуктов ({percentage:.1f}%), {revenue_share:.1f}% выручки")

    if 'XYZ_Category' in product_sales.columns:
        print("\n📊 РАСПРЕДЕЛЕНИЕ XYZ-КАТЕГОРИЙ:")
        xyz_counts = product_sales['XYZ_Category'].value_counts()
        for category, count in xyz_counts.items():
            percentage = (count / len(product_sales)) * 100
            print(f"   {category}: {count} продуктов ({percentage:.1f}%)")

    return product_sales

# 6. ОСНОВНАЯ ФУНКЦИЯ


def main():
    """Основная функция выполнения анализа"""

    print("🚀 ЗАПУСК АНАЛИЗА ДАННЫХ О ПРОДАЖАХ")
    print("=" * 60)

    # 1. ПОДГОТОВКА ДАННЫХ
    print("\n1. 🔄 ПОДГОТОВКА ДАННЫХ")
    customer_df, product_df, sales_df, territories_df = load_data()

    if customer_df is None:
        return

    # Объединение данных
    merged_df = merge_data(customer_df, product_df, sales_df, territories_df)

    # Очистка и преобразование
    cleaned_df = clean_data(merged_df)

    # EDA
    explored_df = perform_eda(cleaned_df)

    # Визуализации EDA
    create_eda_visualizations(explored_df)

    # Сохранение очищенных данных
    save_cleaned_data(explored_df)

    # 2. ВИЗУАЛИЗАЦИЯ И АНАЛИЗ
    print("\n2. 📊 ВИЗУАЛИЗАЦИЯ И АНАЛИЗ")
    create_dashboards(explored_df)
    insights = extract_insights(explored_df)

    # 3. ОБОГАЩЕНИЕ ДАННЫХ
    print("\n3. 🎯 ОБОГАЩЕНИЕ ДАННЫХ")
    rfm_results = perform_rfm_analysis(explored_df)
    abc_xyz_results = perform_abc_xyz_analysis(explored_df)

    # 4. ЗАКЛЮЧЕНИЕ
    print("\n4. 📝 ЗАКЛЮЧЕНИЕ")
    print("=" * 40)

    # Генерация финального заключения
    def generate_final_conclusion(insights, rfm_results, abc_xyz_results):
        """Генерация финального заключения"""

        # Анализ RFM результатов
        champions_count = len(rfm_results[rfm_results['Segment'] == 'Champions'])
        hibernating_count = len(rfm_results[rfm_results['Segment'] == 'Hibernating'])

        # Анализ ABC результатов
        a_products_revenue = abc_xyz_results[abc_xyz_results['ABC_Category'] == 'A']['SalesAmount'].sum()
        total_revenue = abc_xyz_results['SalesAmount'].sum()
        a_percentage = (a_products_revenue / total_revenue) * 100

        conclusion = f"""
        🎯 КЛЮЧЕВЫЕ ВЫВОДЫ:

        1. 📈 ЭФФЕКТИВНОСТЬ ПРОДАЖ:
           {insights[0]}
           {insights[3]}

        2. 👥 КЛИЕНТСКАЯ БАЗА:
           • {champions_count} клиентов в сегменте "Champions" (наиболее ценные)
           • {hibernating_count} клиентов в сегменте "Hibernating" (требуют реактивации)
           {insights[4] if len(insights) > 4 else ''}

        3. 📦 ПРОДУКТОВЫЙ ПОРТФЕЛЬ:
           • Категория A продуктов приносит {a_percentage:.1f}% общей выручки
           • Оптимизация ассортимента может повысить эффективность

        4. 🌍 ГЕОГРАФИЧЕСКОЕ РАСПРЕДЕЛЕНИЕ:
           {insights[5] if len(insights) > 5 else '• Выявлены регионы с наибольшим потенциалом'}

        💡 РЕКОМЕНДАЦИИ:

        1. 🎯 ДЛЯ МАРКЕТИНГА:
           • Разработать программу лояльности для "Champions"
           • Провести реактивационную кампанию для "Hibernating"
           • Персонализировать коммуникации по RFM-сегментам

        2. 📦 ДЛЯ ПРОДАЖ:
           • Сфокусироваться на продвижении продуктов категории A
           • Оптимизировать запасы по результатам XYZ-анализа
           • Разработать кросс-селлинг стратегию

        3. 📊 ДЛЯ БИЗНЕСА:
           • Увеличить присутствие в наиболее прибыльных регионах
           • Мониторить сезонные patterns для планирования
           • Внедрить регулярный RFM и ABC-XYZ анализ
        """

        print(conclusion)

        # Сохранение отчета
        report_content = f"""
    ОТЧЕТ ПО АНАЛИЗУ ПРОДАЖ
    {'=' * 50}

    КЛЮЧЕВЫЕ ИНСАЙТЫ:
    {chr(10).join(['• ' + insight for insight in insights])}

    {conclusion}

    ДАННЫЕ ДЛЯ УГЛУБЛЕННОГО АНАЛИЗА:
    • RFM сегменты сохранены в rfm_results
    • ABC-XYZ анализ сохранен в abc_xyz_results
    • Очищенные данные в cleaned_sales_data.csv
    """

        with open('analysis_final_report.txt', 'w', encoding='utf-8') as f:
            f.write(report_content)

        print("✅ Финальный отчет сохранен в 'analysis_final_report.txt'")

    generate_final_conclusion(insights, rfm_results, abc_xyz_results)

    print("\n✅ АНАЛИЗ ЗАВЕРШЕН!")
    print("📁 Результаты сохранены в файлах:")
    print("   - cleaned_sales_data.csv")
    print("   - analysis_final_report.txt")
    print("   - eda_analysis.png")
    print("   - RFM и ABC-XYZ датафреймы доступны для дальнейшего анализа")
