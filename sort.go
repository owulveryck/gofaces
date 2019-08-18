package gofaces

import "sort"

type byProba []Element

func (b byProba) Len() int           { return len(b) }
func (b byProba) Swap(i, j int)      { b[i], b[j] = b[j], b[i] }
func (b byProba) Less(i, j int) bool { return b[i].Prob < b[j].Prob }

type byConfidence []Box

func (b byConfidence) Len() int           { return len(b) }
func (b byConfidence) Swap(i, j int)      { b[i], b[j] = b[j], b[i] }
func (b byConfidence) Less(i, j int) bool { return b[i].Confidence < b[j].Confidence }

func getOrderedElements(input []float64) []Element {
	elems := make([]Element, len(input))
	for i := 0; i < len(elems); i++ {
		elems[i] = Element{
			Prob:  input[i],
			Class: classes[i],
		}
	}
	sort.Sort(sort.Reverse(byProba(elems)))
	return elems
}
