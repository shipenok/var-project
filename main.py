import yfinance as yf
from calculations import PortfolioVaRCalculator, ProfitCalculator
from visualization import show_visualization_menu

def get_stock_profits(ticker: str, period: str = "1y") -> list:
    """Получение доходностей акции"""
    stock = yf.Ticker(ticker)
    hist = stock.history(period=period)
    
    if hist.empty:
        print(f"Нет данных для {ticker}")
        return []
    
    prices = hist['Close'].tolist()
    
    profits = ProfitCalculator.calculate_profits_arr(prices)

    print(f"{ticker}: {len(profits)} доходностей")
    return profits

def calculate_portfolio_var(tickers: list, weights: list, portfolio_value: int, 
                           confidence: float = 0.99, horizon: int = 7, period: str = "1y"):
    """Расчет VaR портфеля с возможностью визуализации"""
    print("РАСЧЕТ VaR ПОРТФЕЛЯ")
    print("="*40)
    print(f"Портфель: {tickers}")
    print(f"Веса: {[f'{w:.1%}' for w in weights]}")
    print(f"Стоимость: {portfolio_value:,}")
    print(f"Доверие: {confidence:.0%}")
    print(f"Период: {horizon} дней")
    print("="*40)
    
    profits_data = {}
    weights_dict = {}
    
    for ticker, weight in zip(tickers, weights):
        profits = get_stock_profits(ticker, period)
        if profits:
            profits_data[ticker] = profits
            weights_dict[ticker] = weight
    
    if not profits_data:
        print("Не удалось загрузить данные")
        return None    

    calc = PortfolioVaRCalculator(confidence_level=confidence, time_horizon=horizon)
    
    parametric_result = calc.parametric_portfolio_var(profits_data, weights_dict, portfolio_value)
    print("\nПАРАМЕТРИЧЕСКИЙ VaR:")
    print(f"VaR портфеля: ${parametric_result['var_absolute']:,.0f}")
    print(f"VaR в %: {parametric_result['var_percentage']*100:.3f}%")
    print(f"Метод: {parametric_result['method']}")
    
    historical_result = calc.historical_portfolio_var(profits_data, weights_dict, portfolio_value)
    print("\nИСТОРИЧЕСКИЙ VaR:")
    print(f"VaR портфеля: ${historical_result['var_absolute']:,.0f}")
    print(f"VaR в %: {historical_result['var_percentage']*100:.3f}%")
    print(f"Метод: {historical_result['method']}")
    
    print(f"Портфельная доходность: {parametric_result['portfolio_mean']*100:.4f}% (дневная)")
    print(f"Портфельная волатильность: {parametric_result['portfolio_std']*100:.4f}% (дневная)")
    
    print("\n" + "="*60)
    show_graphs = input("Хотите посмотреть графики и диаграммы анализа? (y/n): ").lower().strip()
    
    if show_graphs in ['y', 'yes']:
        show_visualization_menu(
            parametric_result=parametric_result,
            historical_result=historical_result,
            profits_data=profits_data,
            weights=weights_dict,
            confidence_level=confidence,
            portfolio_value=portfolio_value
        )
    
    return {
        'parametric': parametric_result,
        'historical': historical_result,
        'profits_data': profits_data,
        'weights': weights_dict
    }

def portfolio():
    """Основная функция создания и анализа портфеля"""
    print("\nСОЗДАНИЕ ПОРТФЕЛЯ")
    print("="*40)

    tickers_input = input("Введите тикеры через запятую (например: AAPL, MSFT, GOOGL): ")
    tickers = [t.strip().upper() for t in tickers_input.split(',')]
    
    weights = []
    for ticker in tickers:
        weight = float(input(f"Вес для {ticker} (от 0 до 1): "))
        weights.append(weight)
    
    total = sum(weights)
    if abs(total - 1.0) > 0.01:
        print(f"Сумма весов {total:.3f}, нормализую до 1.0")
        weights = [w/total for w in weights]
    
    amount = int(input("Сумма портфеля: "))
    
    confidence = float(input("Уровень доверия (от 0 до 1): "))
    horizon = int(input("Временной горизонт VaR (дни, например 7): "))
    
    print("\nДоступные периоды: 1mo, 3mo, 6mo, 1y, 2y, 5y")
    period = input("Исторический период для данных (по умолчанию 1y): ").strip()
    if not period:
        period = "1y"
    
    results = calculate_portfolio_var(tickers, weights, amount, confidence, horizon, period)
    
    return results


results = portfolio()
