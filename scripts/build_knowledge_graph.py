"""从医疗 JSON 数据构建 Neo4j 知识图谱和 NER 实体词表。"""

from __future__ import annotations

import os
import re
import json
import logging
import py2neo
from tqdm import tqdm
import argparse

from medical_rag.core.config import settings

logger = logging.getLogger(__name__)


#导入普通实体
def import_entity(client,type,entity):
    def create_node(client,type,name):
        order = """create (n:%s{名称:"%s"})"""%(type,name)
        client.run(order)

    logger.info('正在导入 %s 类数据 (%d 个)', type, len(entity))
    for en in tqdm(entity):
        create_node(client,type,en)
#导入疾病类实体
def import_disease_data(client,type,entity):
    logger.info('正在导入 %s 类数据 (%d 个)', type, len(entity))
    for disease in tqdm(entity):
        node = py2neo.Node(type,
                           名称=disease["名称"],
                           疾病简介=disease["疾病简介"],
                           疾病病因=disease["疾病病因"],
                           预防措施=disease["预防措施"],
                           治疗周期=disease["治疗周期"],
                           治愈概率=disease["治愈概率"],
                           疾病易感人群=disease["疾病易感人群"],

                           )
        client.create(node)

def create_all_relationship(client,all_relationship):
    def create_relationship(client,type1, name1,relation, type2,name2):
        order = """match (a:%s{名称:"%s"}),(b:%s{名称:"%s"}) create (a)-[r:%s]->(b)"""%(type1,name1,type2,name2,relation)
        client.run(order)
    print("正在导入关系.....")
    for type1, name1,relation, type2,name2  in tqdm(all_relationship):
        create_relationship(client,type1, name1,relation, type2,name2)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format='[%(asctime)s] %(levelname)s %(name)s: %(message)s')
    #连接数据库的一些参数
    parser = argparse.ArgumentParser(description="通过medical.json文件,创建一个知识图谱")
    parser.add_argument('--website', type=str, default="bolt://localhost:7687", help='neo4j的连接网站')
    parser.add_argument('--user', type=str, default=settings.NEO4J_USER, help='neo4j的用户名')
    parser.add_argument('--password', type=str, default=settings.NEO4J_PASSWORD, help='neo4j的密码')
    parser.add_argument('--dbname', type=str, default=settings.NEO4J_DBNAME, help='数据库名称')
    args = parser.parse_args()

    #连接...
    client = py2neo.Graph(args.website, user=args.user, password=args.password, name=args.dbname)

    #将数据库中的内容删光
    is_delete = input('注意:是否删除neo4j上的所有实体 (y/n):')
    if is_delete=='y':
        client.run("match (n) detach delete (n)")

    with open('./data/medical_new_2.json','r',encoding='utf-8') as f:
        all_data = f.read().split('\n')
    
    #所有实体
    all_entity = {
        "疾病": [],
        "药品": [],
        "食物": [],
        "检查项目":[],
        "科目":[],
        "疾病症状":[],
        "治疗方法":[],
        "药品商":[],
    }
    
    # 实体间的关系
    relationship = []
    for i,data in enumerate(all_data):
        if (len(data) < 3):
            continue
        # rstrip + 去除末尾逗号 + json.loads。
        line = data.rstrip().rstrip(',')
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            logger.warning("跳过无法解析的 JSON 行: %s", line[:80])
            continue

        disease_name = data.get("name","")
        all_entity["疾病"].append({
            "名称":disease_name,
            "疾病简介": data.get("desc", ""),
            "疾病病因": data.get("cause", ""),
            "预防措施": data.get("prevent", ""),
            "治疗周期":data.get("cure_lasttime",""),
            "治愈概率": data.get("cured_prob", ""),
            "疾病易感人群": data.get("easy_get", ""),
        })

        drugs = data.get("common_drug", []) + data.get("recommand_drug", [])
        all_entity["药品"].extend(drugs)  # 添加药品实体
        if drugs:
            relationship.extend([("疾病", disease_name, "疾病使用药品", "药品",durg)for durg in drugs])

        do_eat = data.get("do_eat",[])+data.get("recommand_eat",[])
        no_eat = data.get("not_eat",[])
        all_entity["食物"].extend(do_eat+no_eat)
        if do_eat:
            relationship.extend([("疾病", disease_name,"疾病宜吃食物","食物",f) for f in do_eat])
        if no_eat:
            relationship.extend([("疾病", disease_name, "疾病忌吃食物", "食物", f) for f in no_eat])

        check = data.get("check", [])
        all_entity["检查项目"].extend(check)
        if check:
            relationship.extend([("疾病", disease_name, "疾病所需检查", "检查项目",ch) for ch in check])

        cure_department=data.get("cure_department", [])
        all_entity["科目"].extend(cure_department)
        if cure_department:
            relationship.append(("疾病", disease_name, "疾病所属科目", "科目",cure_department[-1]))

        symptom = data.get("symptom",[])
        # 清洗症状名末尾的省略号（数据集偶尔有 'XXX症状...' 这种值）。
        symptom = [s[:-3] if isinstance(s, str) and s.endswith('...') else s for s in symptom]
        all_entity["疾病症状"].extend(symptom)
        if symptom:
            relationship.extend([("疾病", disease_name, "疾病的症状", "疾病症状",sy )for sy in symptom])

        cure_way = data.get("cure_way", [])
        if cure_way:
            # glm 处理数据集偶尔有格式错误：把字符串包成 list，这里展平
            cure_way = [c[0] if isinstance(c, list) else c for c in cure_way]
            cure_way = [s for s in cure_way if isinstance(s, str) and len(s) >= 2]
            all_entity["治疗方法"].extend(cure_way)
            relationship.extend([("疾病", disease_name, "治疗的方法", "治疗方法", cure_w) for cure_w in cure_way])
            

        acompany_with = data.get("acompany", [])
        if acompany_with:
            relationship.extend([("疾病", disease_name, "疾病并发疾病", "疾病", disease) for disease in acompany_with])

        drug_detail = data.get("drug_detail",[])
        for detail in drug_detail:
            lis = detail.split(',')
            if(len(lis)!=2):
                continue
            p,d = lis[0],lis[1]
            all_entity["药品商"].append(d)
            all_entity["药品"].append(p)
            relationship.append(('药品商',d,"生产","药品",p))
    for i in range(len(relationship)):
        if len(relationship[i])!=5:
            print(relationship[i])
    relationship = list(set(relationship))
    all_entity = {k:(list(set(v)) if k!="疾病" else v)for k,v in all_entity.items()}
    
    # 保存关系 放到data下
    with open("./data/rel_aug.txt",'w',encoding='utf-8') as f:
        for rel in relationship:
            f.write(" ".join(rel))
            f.write('\n')

    if not os.path.exists('data/ent_aug'):
        os.mkdir('data/ent_aug')
    for k,v in all_entity.items():
        with open(f'data/ent_aug/{k}.txt','w',encoding='utf8') as f:
            if(k!='疾病'):
                for i,ent in enumerate(v):
                    f.write(ent+('\n' if i != len(v)-1 else ''))
            else:
                for i,ent in enumerate(v):
                    f.write(ent['名称']+('\n' if i != len(v)-1 else ''))

    #将属性和实体导入到neo4j上,注:只有疾病有属性，特判
    for k in all_entity:
        if k!="疾病":
            import_entity(client,k,all_entity[k])
        else:
            
            import_disease_data(client,k,all_entity[k])
    create_all_relationship(client,relationship)

    

    

    
