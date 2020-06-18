from datasets import build_lm_wider_face


dataset = build_lm_wider_face('train')
item = dataset.__getitem__(0)