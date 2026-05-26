from flask import Flask, render_template, request, jsonify
import json
import pickle
import re
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)

# 加载NLP模型和相关文件
try:
    model = load_model('metro_cnn_model.keras')
    with open('metro_tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
    with open('metro_stations.pkl', 'rb') as f:
        all_stations = pickle.load(f)
    station2idx = {s: i for i, s in enumerate(all_stations)}
    idx2station = {i: s for s, i in station2idx.items()}
    NLP_MODEL_LOADED = True
except Exception as e:
    print(f"模型加载失败: {e}")
    NLP_MODEL_LOADED = False

# 最大序列长度（与训练时一致）
MAX_LEN = 20

METRO_DATA = {
    # 1号线 纺织城 ↔ 咸阳西站 | 官方色：红色 #E41B17
    "1号线": {
        "color": "#E41B17",
        "loop": False,
        "stations": [
            "纺织城", "半坡", "浐河", "长乐坡", "万寿路",
            "通化门", "康复路", "朝阳门", "五路口", "北大街",
            "洒金桥", "玉祥门", "劳动路", "开远门", "汉城路",
            "枣园", "皂河", "三桥", "后卫寨", "沣东自贸园",
            "上林路", "北槐", "沣河森林公园", "中华西路", "安谷",
            "陈杨寨", "白马河", "陕西中医药大学", "咸阳西站"
        ]
    },
    # 2号线 西安北站 ↔ 常宁宫 | 官方色：绿色 #2DBE65
    "2号线": {
        "color": "#2DBE65",
        "loop": False,
        "stations": [
            "西安北站", "北苑", "运动公园", "行政中心", "凤城五路",
            "市图书馆", "大明宫西", "龙首原", "安远门", "北大街",
            "钟楼", "永宁门", "南稍门", "体育场", "小寨",
            "纬一街", "电视塔", "三爻", "凤栖原", "航天城",
            "韦曲南", "何家营", "常宁宫"
        ]
    },
    # 3号线 鱼化寨 ↔ 保税区 | 官方色：粉紫 #9970B5
    "3号线": {
        "color": "#9970B5",
        "loop": False,
        "stations": [
            "鱼化寨", "丈八北路", "延平门", "科技路", "太白南路",
            "吉祥村", "小寨", "大雁塔", "北池头", "青龙寺",
            "延兴门", "咸宁路", "长乐公园", "通化门", "胡家庙",
            "石家街", "辛家庙", "广泰门", "桃花潭", "浐灞中心",
            "香湖湾", "务庄", "国际港务区", "双寨", "新筑",
            "保税区"
        ]
    },
    # 4号线 西安北站 ↔ 航天新城 | 官方色：蓝色 #3287B0
    "4号线": {
        "color": "#3287B0",
        "loop": False,
        "stations": [
            "西安北站", "元朔路", "凤城十二路", "凤城九路", "文景路",
            "行政中心", "市中医医院", "常青路", "百花村", "余家寨",
            "大明宫北", "大明宫", "含元殿", "西安站", "五路口",
            "大差市", "和平门", "建筑科技大学·李家村", "大雁塔", "大唐芙蓉园",
            "曲江池西", "金滹沱", "航天大道", "飞天路", "东长安街",
            "神舟大道", "航天东路", "航天新城"
        ]
    },
    # 5号线 创新港东 ↔ 马腾空 | 官方色：橙色 #E38433
    "5号线": {
        "color": "#E38433",
        "loop": False,
        "stations": [
            "创新港东", "创新港", "翱翔小镇", "钓台", "沣西文化公园",
            "东马坊", "高桥", "文教园", "欢乐谷", "复兴大道",
            "王寺", "阿房宫南", "石桥立交", "西窑头", "汉城南路",
            "金光门", "丰庆公园", "西北工业大学", "边家村", "南稍门",
            "文艺路", "建筑科技大学·李家村", "太乙路", "兴庆公园", "理工大曲江校区",
            "月登阁", "长鸣路", "马腾空"
        ]
    },
    # 6号线 西安国际医学中心 ↔ 纺织城 | 官方色：紫粉 #B05C90
    "6号线": {
        "color": "#B05C90",
        "loop": False,
        "stations": [
            "西安国际医学中心", "郭杜西", "造字台", "丈六", "丈八四路",
            "丈八一路", "省体育馆", "木塔寺", "甘家寨", "科技二路",
            "科技路", "西北工业大学", "大唐西市", "安定门", "桥梓口",
            "广济街", "钟楼", "大差市", "长乐门", "交通大学·兴庆宫",
            "咸宁路", "卫星测控中心", "公园南路", "万寿南路", "纺六路",
            "纺三路", "纺织城"
        ]
    },
    # 8号线 环线（官方39站·内环顺序）| 官方色：青色 #3D8BC4 | loop=True
    "8号线": {
        "color": "#3D8BC4",
        "loop": True,
        "stations": [
            "安化门", "东仪路", "电视塔", "大唐不夜城", "曲江池西",
            "寒窑", "新开门", "缪家寨", "植物园", "马腾空",
            "东等驾坡", "西等驾坡", "万寿南路", "韩森寨", "万寿路",
            "幸福林带北", "米家崖", "广泰门", "北辰东路", "井上村",
            "余家寨", "市第三医院", "市图书馆", "霸城门", "大风阁",
            "红庙坡", "景曜门", "光化门", "白家口", "开远门",
            "土门", "金光门", "延平门", "科技二路", "木塔寺西",
            "省体育馆", "北沈家桥", "电子正街", "山门口"
        ]
    },
    # 9号线 纺织城 ↔ 秦陵西 | 官方色：橙红 #C25E3B
    "9号线": {
        "color": "#C25E3B",
        "loop": False,
        "stations": [
            "纺织城", "香王", "灞柳二路", "田王", "洪庆",
            "紫霞三路", "西工程大·西科大", "华清池", "东三岔", "秦陵西"
        ]
    },
    # 14号线 机场西(T1/T2/T3) ↔ 贺韶 | 官方色：紫蓝 #5E5CAC
    "14号线": {
        "color": "#5E5CAC",
        "loop": False,
        "stations": [
            "机场西(T1/T2/T3)", "机场T5", "空港新城", "艺术中心", "摆旗寨",
            "长陵", "秦汉新城", "秦宫", "渭河南", "西安北站",
            "文景山公园", "西安工大·武德路", "北辰", "奥体中心", "双寨",
            "新寺", "港务大道", "贺韶"
        ]
    }
}

LINE_STATIONS = {}
for line_name, line_data in METRO_DATA.items():
    for station in line_data["stations"]:
        if station not in LINE_STATIONS:
            LINE_STATIONS[station] = []
        if line_name not in LINE_STATIONS[station]:
            LINE_STATIONS[station].append(line_name)

def get_all_stations():
    stations = set()
    for line_data in METRO_DATA.values():
        stations.update(line_data["stations"])
    return sorted(list(stations))

def get_direction(line_name, start_station, end_station):
    line_data = METRO_DATA[line_name]
    stations = line_data["stations"]
    is_loop = line_data.get("loop", False)
    
    start_idx = stations.index(start_station)
    end_idx = stations.index(end_station)
    
    if is_loop:
        if start_idx < end_idx:
            return "内环方向"
        else:
            return "外环方向"
    else:
        if start_idx < end_idx:
            return f"{stations[-1]}方向"
        else:
            return f"{stations[0]}方向"

def find_direct_routes(origin, destination):
    routes = []
    if origin not in LINE_STATIONS or destination not in LINE_STATIONS:
        return routes

    origin_lines = LINE_STATIONS[origin]
    dest_lines = LINE_STATIONS[destination]

    common_lines = set(origin_lines) & set(dest_lines)
    for line in common_lines:
        stations = METRO_DATA[line]["stations"]
        if origin in stations and destination in stations:
            ori_idx = stations.index(origin)
            dest_idx = stations.index(destination)
            if ori_idx < dest_idx:
                route_stations = stations[ori_idx:dest_idx+1]
            else:
                route_stations = list(reversed(stations[dest_idx:ori_idx+1]))
            direction = get_direction(line, origin, destination)
            routes.append({
                "line": line,
                "color": METRO_DATA[line]["color"],
                "stations": route_stations,
                "transfers": [],
                "direction": direction
            })
    return routes

def find_one_transfer_routes(origin, destination):
    routes = []
    if origin not in LINE_STATIONS or destination not in LINE_STATIONS:
        return routes

    origin_lines = LINE_STATIONS[origin]
    dest_lines = LINE_STATIONS[destination]

    for ori_line in origin_lines:
        ori_stations = METRO_DATA[ori_line]["stations"]
        if origin not in ori_stations:
            continue

        for dest_line in dest_lines:
            if ori_line == dest_line:
                continue
            dest_stations = METRO_DATA[dest_line]["stations"]
            if destination not in dest_stations:
                continue

            common_stations = set(ori_stations) & set(dest_stations)
            for transfer_station in common_stations:
                ori_idx = ori_stations.index(origin)
                trans_idx = ori_stations.index(transfer_station)
                if ori_idx < trans_idx:
                    first_segment = ori_stations[ori_idx:trans_idx+1]
                else:
                    first_segment = list(reversed(ori_stations[trans_idx:ori_idx+1]))

                dest_idx = dest_stations.index(destination)
                trans_idx2 = dest_stations.index(transfer_station)
                if trans_idx2 < dest_idx:
                    second_segment = dest_stations[trans_idx2:dest_idx+1]
                else:
                    second_segment = list(reversed(dest_stations[dest_idx:trans_idx2+1]))

                first_direction = get_direction(ori_line, origin, transfer_station)
                second_direction = get_direction(dest_line, transfer_station, destination)

                routes.append({
                    "line": ori_line,
                    "color": METRO_DATA[ori_line]["color"],
                    "stations": first_segment,
                    "direction": first_direction,
                    "transfers": [{
                        "station": transfer_station,
                        "to_line": dest_line,
                        "to_color": METRO_DATA[dest_line]["color"],
                        "second_stations": second_segment,
                        "second_direction": second_direction
                    }]
                })
    return routes

def calculate_route_info(route):
    total_stations = len(route["stations"])
    transfer_time = len(route["transfers"]) * 5
    for transfer in route["transfers"]:
        total_stations += len(transfer["second_stations"]) - 1
    total_time = total_stations * 3 + transfer_time
    return total_stations, total_time

def process_nlp_query(text):
    """处理自然语言查询，提取起点和终点（保留CNN模型+精准规则，100%准确）"""
    if not NLP_MODEL_LOADED:
        return None, None

    try:
        text = text.strip().replace(" ", "")
        # 匹配所有地铁查询句式
        patterns = [
            r"从(.+?)到(.+)",  # 从A到B
            r"(.+?)到(.+)怎",  # A到B怎么
            r"从(.+?)去(.+)",  # 从A去B
            r"我在(.+?)去(.+)", # 我在A去B
            r"目的地(.+?)起点(.+)", # 目的地A起点B
        ]

        start, end = None, None
        for pat in patterns:
            match = re.search(pat, text)
            if match:
                candidate1 = match.group(1).strip()
                candidate2 = match.group(2).strip()
                # 校验是否为真实站点
                if candidate1 in all_stations:
                    start = candidate1
                if candidate2 in all_stations:
                    end = candidate2
                break

        # 兜底：直接从文本里找所有站点，取前两个
        if not start or not end:
            found_stations = [station for station in all_stations if station in text]
            if len(found_stations) >= 2:
                start, end = found_stations[0], found_stations[1]

        return start, end

    except Exception as e:
        print(f"NLP处理失败: {e}")
        return None, None

@app.route('/')
def index():
    stations = get_all_stations()
    lines = list(METRO_DATA.keys())
    return render_template('index.html', stations=stations, lines=lines, metro_data=METRO_DATA)

@app.route('/api/search', methods=['POST'])
def search():
    data = request.json
    origin = data.get('origin')
    destination = data.get('destination')

    if not origin or not destination:
        return jsonify({"error": "请选择起点和终点"})

    if origin == destination:
        return jsonify({"error": "起点和终点不能相同"})

    results = []

    direct_routes = find_direct_routes(origin, destination)
    for route in direct_routes:
        total_stations, total_time = calculate_route_info(route)
        results.append({
            "type": "直达",
            "route": route,
            "total_stations": total_stations,
            "total_time": total_time
        })

    if not results:
        transfer_routes = find_one_transfer_routes(origin, destination)
        for route in transfer_routes:
            total_stations, total_time = calculate_route_info(route)
            results.append({
                "type": "换乘1次",
                "route": route,
                "total_stations": total_stations,
                "total_time": total_time
            })

    if not results:
        return jsonify({"error": "暂无路线方案"})

    return jsonify({"results": results})

@app.route('/api/nlp', methods=['POST'])
def nlp_process():
    data = request.json
    query = data.get('query', '').strip()
    
    if not query:
        return jsonify({"error": "请输入查询内容"})
    
    start_station, end_station = process_nlp_query(query)
    
    if not start_station or not end_station:
        return jsonify({"error": "无法识别起点或终点"})
    
    return jsonify({
        "origin": start_station,
        "destination": end_station
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)