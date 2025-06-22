import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd


plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

def plot_historical_var_distribution(portfolio_profits, var_result, confidence_level):
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    time_scaling_factor = var_result.get('time_scaling_factor', 1)
    scaled_profits = [r * time_scaling_factor for r in portfolio_profits]
    sorted_profits = sorted(scaled_profits)
    
    ax1.hist(scaled_profits, bins=30, alpha=0.7, color='lightblue', 
             edgecolor='black', density=True)
    
    var_value = var_result['var_percentage']
    ax1.axvline(var_value, color='red', linestyle='--', linewidth=2,
                label=f'VaR ({confidence_level:.0%}): {var_value:.4f}')
    
    alpha = 1 - confidence_level
    x_fill = np.linspace(min(scaled_profits), var_value, 100)
    y_fill = np.zeros_like(x_fill)
    ax1.fill_between(x_fill, y_fill, ax1.get_ylim()[1], alpha=0.3, color='red',
                     label=f'Область риска ({alpha:.0%})')
    
    ax1.set_xlabel('Доходность портфеля')
    ax1.set_ylabel('Плотность')
    ax1.set_title('Распределение доходностей портфеля\n(Historical VaR)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    var_index = int(len(sorted_profits) * alpha)
    
    ax2.plot(range(len(sorted_profits)), sorted_profits, 'b-', linewidth=1.5,
             label='Отсортированные доходности')
    ax2.scatter(range(var_index + 1), sorted_profits[:var_index + 1], 
                color='red', s=8, alpha=0.7, label=f'Отброшенные ({alpha:.0%})')
    ax2.axhline(var_value, color='red', linestyle='--', linewidth=2,
                label=f'VaR: {var_value:.4f}')
    
    ax2.set_xlabel('Позиция в отсортированном ряду')
    ax2.set_ylabel('Доходность')
    ax2.set_title('Отсортированные доходности\n(для Historical VaR)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def plot_correlation_matrix(profits_data):

    if len(profits_data) < 2:
        print("Для корреляционного анализа нужно минимум 2 актива")
        return
    
    min_length = min(len(profits) for profits in profits_data.values())
    df_profits = pd.DataFrame({
        asset: profits[:min_length] 
        for asset, profits in profits_data.items()
    })
    
    corr_matrix = df_profits.corr()
    
    plt.figure(figsize=(10, 8))
    
    sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r', center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": .8},
                fmt='.3f', annot_kws={'size': 10})
    
    plt.title('Корреляционная матрица доходностей активов', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

def plot_individual_returns(profits_data, weights):

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    for asset, profits in profits_data.items():
        ax1.plot(range(len(profits)), profits, label=f'{asset} (вес: {weights[asset]:.1%})',
                alpha=0.8, linewidth=1.2)
    
    ax1.set_xlabel('Период (дни)')
    ax1.set_ylabel('Дневная доходность')
    ax1.set_title('Временные ряды доходностей активов портфеля')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    for asset, profits in profits_data.items():
        cumulative_returns = np.cumprod(1 + np.array(profits)) - 1
        ax2.plot(range(len(cumulative_returns)), cumulative_returns * 100,
                label=f'{asset}', alpha=0.8, linewidth=1.5)
    
    ax2.set_xlabel('Период (дни)')
    ax2.set_ylabel('Кумулятивная доходность (%)')
    ax2.set_title('Кумулятивные доходности активов')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def plot_portfolio_performance(profits_data, weights):

    assets = list(profits_data.keys())
    n_periods = len(profits_data[assets[0]])
    
    current_portfolio_returns = []
    for t in range(n_periods):
        period_return = sum(weights[asset] * profits_data[asset][t] for asset in assets)
        current_portfolio_returns.append(period_return)
    
    equal_weights = {asset: 1.0/len(assets) for asset in assets}
    equal_portfolio_returns = []
    for t in range(n_periods):
        period_return = sum(equal_weights[asset] * profits_data[asset][t] for asset in assets)
        equal_portfolio_returns.append(period_return)
    
    current_cumulative = np.cumprod(1 + np.array(current_portfolio_returns)) - 1
    equal_cumulative = np.cumprod(1 + np.array(equal_portfolio_returns)) - 1
    
    plt.figure(figsize=(14, 8))
    
    plt.plot(range(len(current_cumulative)), current_cumulative * 100,
             label='Текущий портфель', linewidth=2, color='blue')
    plt.plot(range(len(equal_cumulative)), equal_cumulative * 100,
             label='Равновесный портфель', linewidth=2, color='orange', linestyle='--')
    
    plt.xlabel('Период (дни)')
    plt.ylabel('Кумулятивная доходность (%)')
    plt.title('Сравнение производительности портфелей')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    current_total_return = (current_cumulative[-1]) * 100
    equal_total_return = (equal_cumulative[-1]) * 100
    
    plt.tight_layout()
    plt.show()


def plot_var_timeline(profits_data, weights, confidence_level, portfolio_value, window_size=60):

    from calculations import PortfolioVaRCalculator
    
    assets = list(profits_data.keys())
    n_periods = len(profits_data[assets[0]])
    
    portfolio_returns = []
    for t in range(n_periods):
        period_return = sum(weights[asset] * profits_data[asset][t] for asset in assets)
        portfolio_returns.append(period_return)
    
    if len(portfolio_returns) < window_size:
        print(f"Недостаточно данных для окна {window_size} дней. Доступно {len(portfolio_returns)} дней.")
        return

    historical_var_timeline = []
    parametric_var_timeline = []
    dates = []
    
    for i in range(window_size, len(portfolio_returns)):
        window_returns = portfolio_returns[i-window_size:i]
        
        calc = PortfolioVaRCalculator(confidence_level=confidence_level, time_horizon=7)
        
        hist_result = calc.var_calculator.historical_var(window_returns, portfolio_value)
        historical_var_timeline.append(hist_result['var_absolute'])

        param_result = calc.var_calculator.parametric_var(window_returns, portfolio_value)
        parametric_var_timeline.append(param_result['var_absolute'])
        
        dates.append(i)
    
    plt.figure(figsize=(15, 8))
    
    plt.plot(dates, historical_var_timeline, linewidth=2, color='red', alpha=0.8, 
             label='Historical VaR')
    plt.plot(dates, parametric_var_timeline, linewidth=2, color='orange', alpha=0.8,
             label='Parametric VaR')
    
    plt.fill_between(dates, historical_var_timeline, alpha=0.2, color='red')
    
    plt.xlabel('Период (дни)')
    plt.ylabel('VaR (USD)')
    plt.title(f'Изменение VaR во времени (сравнение методов)\n'
              f'(скользящее окно {window_size} дней, доверие {confidence_level:.0%})')
    plt.grid(True, alpha=0.3)

    avg_hist_var = np.mean(historical_var_timeline)
    avg_param_var = np.mean(parametric_var_timeline)
    
    plt.axhline(avg_hist_var, color='red', linestyle='--', alpha=0.5, 
                label=f'Средний Historical VaR: ${avg_hist_var:,.0f}')
    plt.axhline(avg_param_var, color='orange', linestyle='--', alpha=0.5,
                label=f'Средний Parametric VaR: ${avg_param_var:,.0f}')

    max_hist_var = np.max(historical_var_timeline)
    min_hist_var = np.min(historical_var_timeline)
    max_param_var = np.max(parametric_var_timeline)
    min_param_var = np.min(parametric_var_timeline)
    
    plt.text(0.02, 0.98, 
             f'Статистика VaR:\n'
             f'Historical VaR:\n'
             f'  Средний: ${avg_hist_var:,.0f}\n'
             f'  Макс: ${max_hist_var:,.0f}\n'
             f'  Мин: ${min_hist_var:,.0f}\n\n'
             f'Parametric VaR:\n'
             f'  Средний: ${avg_param_var:,.0f}\n'
             f'  Макс: ${max_param_var:,.0f}\n'
             f'  Мин: ${min_param_var:,.0f}\n\n'
             f'Разница (ср.): ${avg_hist_var - avg_param_var:,.0f}',
             transform=plt.gca().transAxes, fontsize=9,
             verticalalignment='top', 
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.show()


def show_visualization_menu(parametric_result, historical_result, profits_data, weights, confidence_level, portfolio_value):

    assets = list(profits_data.keys())
    n_periods = len(profits_data[assets[0]])
    portfolio_profits = []
    for t in range(n_periods):
        period_profit = sum(weights[asset] * profits_data[asset][t] for asset in assets)
        portfolio_profits.append(period_profit)
    
    while True:
        print("\n" + "="*60)
        print("МЕНЮ ВИЗУАЛИЗАЦИИ VaR АНАЛИЗА")
        print("="*60)
        print("1. Распределение доходностей и Historical VaR")
        print("2. Корреляционная матрица активов")
        print("3. Временные ряды доходностей активов")
        print("4. Сравнение с равновесным портфелем")
        print("5. VaR во времени (скользящее окно)")
        print("6. Показать все графики")
        print("0. Выход")
        print("="*60)
        
        try:
            choice = input("Выберите график (0-6): ").strip()
            
            if choice == '1':
                print("Показываю распределение доходностей и Historical VaR...")
                plot_historical_var_distribution(portfolio_profits, historical_result, confidence_level)
            elif choice == '2':
                print("Показываю корреляционную матрицу...")
                plot_correlation_matrix(profits_data)
            elif choice == '3':
                print("Показываю временные ряды доходностей...")
                plot_individual_returns(profits_data, weights)
            elif choice == '4':
                print("Показываю сравнение с равновесным портфелем...")
                plot_portfolio_performance(profits_data, weights)
            elif choice == '5':
                try:
                    window = input("Размер скользящего окна (по умолчанию 60): ").strip()
                    window_size = int(window) if window else 60
                    print(f"Показываю VaR во времени (окно {window_size} дней)...")
                    plot_var_timeline(profits_data, weights, confidence_level, portfolio_value, window_size)
                except ValueError:
                    print("Неверный размер окна, использую 60 дней")
                    plot_var_timeline(profits_data, weights, confidence_level, portfolio_value, 60)
            elif choice == '6':
                print("Показываю все графики...")
                plot_historical_var_distribution(portfolio_profits, historical_result, confidence_level)
                plot_correlation_matrix(profits_data)
                plot_individual_returns(profits_data, weights)
                plot_portfolio_performance(profits_data, weights)
                plot_var_timeline(profits_data, weights, confidence_level, portfolio_value)
                print("Все графики показаны!")
            elif choice == '0':
                print("Выход из меню визуализации.")
                break
            else:
                print("Неверный выбор. Попробуйте снова.")
                
        except KeyboardInterrupt:
            print("\n\nВыход из меню визуализации.")
            break
        except Exception as e:
            print(f"Ошибка при создании графика: {e}")
            print("Попробуйте выбрать другой график.")
