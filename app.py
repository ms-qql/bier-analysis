from dotenv import load_dotenv
import os
load_dotenv()
bier_uplink = os.environ['BIER5_AG_UPLINK']
import anvil.server

anvil.server.connect(bier_uplink)
### IMPORTS ############################################################
from modules.auto_load import auto_import_itc_data, auto_import_itc_oi_data, auto_import_augmento_data, store_bier_result
from modules.auto_load_bmp import auto_import_bmp_data
from modules.auto_load_capriole import auto_import_capriole_data
from modules.database import db_test, create_tables, read_table_date_range_cloud, read_table_last_X_cloud, add_row_itc, add_df_itc, add_df_capriole, add_row_capriole, add_row_bmp, add_df_bmp, add_row_bmp2, add_df_bmp2, add_row_manta, add_df_manta,  store_metric_table, delete_table_last_row_cloud, delete_table_last_x_rows_cloud, upload_itc_table
from modules.graphs import update_price_chart, update_strategy_chart, update_norm_chart, update_category_chart
from modules.matrix_bot import send_a_message  
from modules.matrix_strategy import calc_metric, calc_metric_all
from modules.backtest import save_backtest_score, perform_backtest_bier, get_backtest_stats
from modules.datatable import read_categories, load_category_list
# Run the Anvil server wait_forever() inside this async function
anvil.server.wait_forever()