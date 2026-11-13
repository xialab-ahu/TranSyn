from pretrain_exclude_hits_common import run

if __name__ == '__main__':
    # 默认读取同目录的新排除表；可用--hit-file替换，无需再次修改代码。
    # run(
    #     task_name='noncoding-GPN-MSA',
    #     hit_filename='./noncoding-GPN-MSA-noDMq/noncoding.excluded.variant.txt',
    #     output_name='noncoding-GPN-MSA-exclude-DMQ',
    # )
    run(
        task_name='missense-CaLM',
        hit_filename='./missense-CaLM-noDMq/missense.excluded.variant.txt',
        output_name='missense-CaLM-exclude-DMQ',
    )
