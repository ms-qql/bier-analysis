import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.stats import norm

class BitcoinPeakDetector:
    def __init__(self, window_slope=10, hazard_rate=1/50, ewma_lambda=0.2):
        self.window = int(window_slope)
        self.hazard = hazard_rate
        self.lambda_ewma = ewma_lambda
        self.error_history = []
        
    def detect_full_series(self, data_series):
        """
        Vectorized/Efficient implementation of the voting ensemble for the full series.
        Returns Dataframe with signals.
        """
        data = data_series.values
        n = len(data)
        
        # --- Method A: Slope-Based Turning Point ---
        # Rolling polyfit slopes
        # Using vectorized approach for efficiency: slope ~ correlation * std_y / std_x
        # x is fixed (0..window-1), so std_x is const. slope propto covariance.
        # But np.polyfit in loop is also fine for N=2000. Let's stick closer to logic or use rolling apply.
        
        slopes = pd.Series(data).rolling(window=self.window).apply(
            lambda y: np.polyfit(np.arange(len(y)), y, 1)[0], raw=True
        ).values
        
        # Detect sign changes in slope
        # Peak: Slope goes Positive -> Negative
        # Valley: Slope goes Negative -> Positive
        slope_sign = np.sign(slopes)
        slope_diff = np.diff(slope_sign, prepend=0)
        
        # peak_a[i] = 1 if peak detected at i, -1 if valley, 0 otherwise
        peak_a = np.zeros(n)
        conf_a = np.zeros(n)
        
        # Identify indices where slope changed
        # Note: slope is calculated on window [i-w+1 : i]. Change sign at i means local extremum approx at i-w/2?
        # User skeleton: returns "last_change" index.
        # If slope changes at 'i', it means the trend changed *ending* at 'i'.
        changes_idx = np.where(slope_diff != 0)[0]
        
        for idx in changes_idx:
            if idx < self.window: continue
            strength = abs(slopes[idx])
            normalized_strength = min(strength, 1.0)
            
            # Determine direction
            if slope_sign[idx-1] > 0 and slope_sign[idx] <= 0:
                peak_a[idx] = 1 # Peak
            elif slope_sign[idx-1] < 0 and slope_sign[idx] >= 0:
                peak_a[idx] = -1 # Valley
            
            conf_a[idx] = normalized_strength

        # --- Method B: Bayesian Change Point Detection ---
        # "Shift > threshold"
        # Expanding window for historical? Skeleton: historical = data[:-10], recent=data[-10:]
        # This implies historical grows.
        peak_b = np.zeros(n)
        conf_b = np.zeros(n)
        
        # Simplified expanding window approach
        # For efficiency, we can iterate or use rolling.
        # Let's iterate from window size
        min_history = 30
        window_b = 10
        
        # rolling_mean = pd.Series(data).rolling(window=window_b).mean().values
        # For historical mean/std, we need expanding window up to i-10
        # This is slightly expensive in loop but acceptable.
        
        # Optimization: maintain running sum/sq_sum for historical?
        # Or just use pandas expanding.
        # historical_stats = data.expanding().agg(['mean', 'std']) # shifted by 10
        
        # Let's do a loop for clarity matching skeleton logic
        for i in range(min_history + window_b, n):
            recent = data[i-window_b:i]
            historical = data[:i-window_b]
            
            recent_mean = np.mean(recent)
            hist_mean = np.mean(historical)
            hist_std = np.std(historical)
            
            shift = abs(recent_mean - hist_mean)
            threshold = hist_std * 2 if hist_std > 1e-6 else 1.0
            
            prob = 0.0
            if shift > threshold:
                prob = min(shift / threshold, 1.0)
            
            # Direction? If recent < historical -> Drop (Potential Peak passed?)
            # Logic: If we dropped significantly, we might have passed a peak.
            # Use 'prob' as confidence. Direction inferred from shift.
            direction = 0
            if recent_mean < hist_mean:
                direction = 1 # Peak candidate (shifted down)
            else:
                direction = -1 # Valley candidate (shifted up)
                
            if prob > 0:
                peak_b[i] = direction
                conf_b[i] = prob

        # --- Method C: EWMA Adaptive Threshold ---
        peak_c = np.zeros(n)
        conf_c = np.zeros(n)
        
        ewma = np.zeros(n)
        ewma[0] = data[0]
        # Pandas ewm is faster
        ewma_series = pd.Series(data).ewm(alpha=self.lambda_ewma, adjust=False).mean().values
        
        error = np.abs(data - ewma_series)
        
        # Rolling percentile for threshold
        # Skeleton: self.error_history.append. threshold = percentile(last 30).
        thresholds = pd.Series(error).rolling(window=30).quantile(0.95).values
        
        # Detect
        # If error > threshold, signal.
        # Direction? If data > ewma -> High anomaly (Peak region). Data < ewma -> Low anomaly.
        for i in range(30, n):
            if error[i] > thresholds[i]:
                norm_err = error[i] / (thresholds[i] + 1e-6)
                conf = min(norm_err, 1.0)
                
                if data[i] > ewma_series[i]:
                    peak_c[i] = 1 # Peak region
                else:
                    peak_c[i] = -1 # Valley region
                
                conf_c[i] = conf
                
        # --- Voting ---
        final_peaks = np.zeros(n)
        final_valleys = np.zeros(n)
        final_conf = np.zeros(n)
        
        for i in range(n):
            # Collect votes for Peak (1)
            votes_peak = 0
            confs_peak = []
            if peak_a[i] == 1: votes_peak += 1; confs_peak.append(conf_a[i])
            if peak_b[i] == 1: 
                # Method B needs high prob? Skeleton: "Method B > 0.6 prob"
                if conf_b[i] > 0.6: votes_peak += 1
                confs_peak.append(conf_b[i])
            if peak_c[i] == 1: votes_peak +=1; confs_peak.append(conf_c[i])
            
            # Collect votes for Valley (-1)
            votes_valley = 0
            confs_valley = []
            if peak_a[i] == -1: votes_valley += 1; confs_valley.append(conf_a[i])
            if peak_b[i] == -1: 
                 if conf_b[i] > 0.6: votes_valley += 1
                 confs_valley.append(conf_b[i])
            if peak_c[i] == -1: votes_valley += 1; confs_valley.append(conf_c[i])
            
            # Step 2 Rules
            # Peak detected IF: (Method A = peak) AND (Method B > 0.6 probability) OR (≥2 of 3 methods vote peak)
            is_peak = False
            avg_conf_p = 0
            if (peak_a[i] == 1 and conf_b[i] > 0.6 and peak_b[i] == 1) or votes_peak >= 2:
                is_peak = True
                avg_conf_p = np.mean(confs_peak) if confs_peak else 0

            is_valley = False
            avg_conf_v = 0
            if (peak_a[i] == -1 and conf_b[i] > 0.6 and peak_b[i] == -1) or votes_valley >= 2:
                is_valley = True
                avg_conf_v = np.mean(confs_valley) if confs_valley else 0
                
            if is_peak:
                # Step 3: Quadratic Interpolation (Refinement) - placeholder logic
                # If avg_conf_p > 0.8: refine timestamp. 
                # For now, mark this bar.
                final_peaks[i] = 1
                final_conf[i] = avg_conf_p
                
            if is_valley:
                final_valleys[i] = 1
                final_conf[i] = avg_conf_v
                
        return final_peaks, final_valleys, final_conf
